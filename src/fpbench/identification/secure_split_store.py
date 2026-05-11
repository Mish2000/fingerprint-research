from __future__ import annotations

import hashlib
import json
from contextlib import suppress
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence
from urllib.parse import urlsplit

import numpy as np

from apps.api.method_registry import MethodRegistryError, load_api_method_registry

# Local fallback intentionally omits embedded credentials. Production/demo deployments should set
# DATABASE_URL and, when using the split-DB layout, IDENTITY_DATABASE_URL.
DEFAULT_DATABASE_URL = "postgresql://127.0.0.1:5432/biometric_db"
_PREFIX_RE = re.compile(r"[^a-zA-Z0-9_]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def normalize_national_id(national_id: str) -> str:
    return "".join(ch for ch in str(national_id).strip() if ch.isdigit())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class PersonDirectoryRecord:
    random_id: str
    full_name: str
    name_norm: str
    national_id: str
    created_at: str


@dataclass(frozen=True)
class RawFingerprintRecord:
    random_id: str
    capture: str
    ext: str
    sha256: str
    byte_size: Optional[int]
    created_at: str
    legacy_image_bytes_present: bool = False


@dataclass(frozen=True)
class FeatureVectorRecord:
    random_id: str
    method: str
    dim: int
    created_at: str
    vector: np.ndarray


@dataclass(frozen=True)
class EnrollmentReceipt:
    random_id: str
    created_at: str
    vector_methods: List[str]
    image_sha256: str


@dataclass(frozen=True)
class IdentifyHints:
    name_pattern: Optional[str] = None
    national_id_pattern: Optional[str] = None
    created_from: Optional[str] = None
    created_to: Optional[str] = None


@dataclass(frozen=True)
class VectorSpec:
    method: str
    dim: int
    vector_kind: str
    distance_metric: str
    generic_storage_column: str
    legacy_storage_column: Optional[str] = None
    legacy_storage_enabled: bool = False
    generic_storage_enabled: bool = True
    preferred_storage: str = "generic"

    @property
    def column(self) -> str:
        """Backward-compatible alias for callers that still expect one vector column."""
        return self.legacy_storage_column or self.generic_storage_column


@dataclass(frozen=True)
class _IdentityMapRow:
    random_id: str
    national_id: str
    created_at: datetime
    full_name: str = ""
    name_norm: str = ""


@dataclass(frozen=True)
class _LegacyPersonProfileRow:
    random_id: str
    full_name: str
    name_norm: str
    created_at: datetime


@dataclass(frozen=True)
class _IdentityReconciliationDrift:
    validation_mode: str
    sample_limit: int
    people_count: int
    identity_count: int
    people_without_identity_count: int
    people_without_identity_sample: List[str]
    identity_without_people_count: int
    identity_without_people_sample: List[str]


LEGACY_VECTOR_COLUMN_BY_DIM = {
    512: "vector_512",
    768: "vector_768",
}
GENERIC_VECTOR_COLUMN_BY_DIM = dict(LEGACY_VECTOR_COLUMN_BY_DIM)
LEGACY_COMPATIBLE_RETRIEVAL_METHODS = frozenset({"dl", "vit"})
VECTOR_STORAGE_SCHEMA = {
    "mode": "method_generic_pgvector_table_with_legacy_compat",
    "method_generic_vectors_supported": True,
    "distance_metric": "cosine",
    "legacy_vector_columns_by_dim": {
        str(dim): column for dim, column in sorted(LEGACY_VECTOR_COLUMN_BY_DIM.items())
    },
    "generic_vector_columns_by_dim": {
        str(dim): column for dim, column in sorted(GENERIC_VECTOR_COLUMN_BY_DIM.items())
    },
    "indexed_dimensions_available": sorted(GENERIC_VECTOR_COLUMN_BY_DIM),
    "legacy_compatibility_enabled": True,
    "write_strategy": "generic_authoritative_with_explicit_legacy_compat",
    "search_strategy": "generic_first_with_explicit_legacy_fallback",
}


def _load_vector_specs_from_registry() -> Dict[str, VectorSpec]:
    registry = load_api_method_registry()
    specs: Dict[str, VectorSpec] = {}
    for method, spec in registry.retrieval_vector_specs().items():
        generic_column = GENERIC_VECTOR_COLUMN_BY_DIM.get(int(spec.dim))
        if generic_column is None:
            raise RuntimeError(
                f"retrieval method {method!r} declares vector dim {spec.dim}, but the "
                "method-generic pgvector storage schema has no compatible indexed column."
            )
        legacy_enabled = method in LEGACY_COMPATIBLE_RETRIEVAL_METHODS
        legacy_column = LEGACY_VECTOR_COLUMN_BY_DIM.get(int(spec.dim)) if legacy_enabled else None
        if legacy_enabled and legacy_column is None:
            raise RuntimeError(
                f"retrieval method {method!r} is marked legacy-compatible with dim {spec.dim}, but "
                "feature_vectors has no compatible pgvector column."
            )
        specs[method] = VectorSpec(
            method=method,
            dim=int(spec.dim),
            vector_kind=spec.vector_kind,
            distance_metric=spec.distance_metric,
            generic_storage_column=generic_column,
            legacy_storage_column=legacy_column,
            legacy_storage_enabled=legacy_enabled,
            generic_storage_enabled=True,
            preferred_storage="dual" if legacy_enabled else "generic",
        )
    return specs


def _load_rerank_methods_from_registry() -> frozenset[str]:
    return frozenset(load_api_method_registry().supported_rerank_methods())


VECTOR_SPECS: Dict[str, VectorSpec] = _load_vector_specs_from_registry()
IDENTIFICATION_RETRIEVAL_VECTOR_METHODS = frozenset(VECTOR_SPECS.keys())
IDENTIFICATION_RERANK_METHODS = _load_rerank_methods_from_registry()
INSPECTION_MAX_ID_SCAN = 10_000
RECONCILIATION_SAMPLE_LIMIT = 10
RECONCILIATION_STREAM_BATCH_SIZE = 1_000


def _legacy_storage_methods() -> frozenset[str]:
    return frozenset(
        method
        for method, spec in VECTOR_SPECS.items()
        if spec.legacy_storage_enabled and spec.legacy_storage_column is not None
    )


def _generic_storage_methods() -> frozenset[str]:
    return frozenset(
        method
        for method, spec in VECTOR_SPECS.items()
        if spec.generic_storage_enabled
    )


def _dual_write_methods() -> frozenset[str]:
    return frozenset(
        method
        for method, spec in VECTOR_SPECS.items()
        if spec.generic_storage_enabled and spec.legacy_storage_enabled
    )


def _generic_only_methods() -> frozenset[str]:
    return frozenset(
        method
        for method, spec in VECTOR_SPECS.items()
        if spec.generic_storage_enabled and not spec.legacy_storage_enabled
    )


def unsupported_identification_retrieval_message(method: str) -> str:
    supported = sorted(IDENTIFICATION_RETRIEVAL_VECTOR_METHODS)
    rerank = sorted(IDENTIFICATION_RERANK_METHODS)
    method_norm = str(method).strip().lower()
    try:
        resolved = load_api_method_registry().resolve(method_norm, field_name="retrieval_method")
    except MethodRegistryError:
        resolved = None
    if resolved is not None and resolved.definition.identification_role.rerank_capable:
        reason = (
            resolved.definition.identification_role.retrieval_unavailable_reason
            or "direct vector retrieval is not configured"
        )
        if resolved.definition.is_experimental or resolved.definition.identification_role.experimental:
            detail = (
                f"Method {resolved.canonical_api_name!r} is currently an experimental rerank-only method "
                "and does not have a validated fixed-size direct retrieval vector adapter yet."
            )
            if resolved.definition.identification_role.future_adapter_hint:
                detail = f"{detail} {resolved.definition.identification_role.future_adapter_hint}"
            return (
                f"{detail} Use one of {supported} for direct vector retrieval and optionally rerank "
                f"the shortlist with {rerank}."
            )
        return (
            f"retrieval_method={method!r} resolves to {resolved.canonical_api_name!r}, which is rerank-only "
            f"for 1:N identification because {reason}. Use one of {supported} for direct vector retrieval "
            f"and optionally rerank the shortlist with {rerank}."
        )
    return (
        f"retrieval_method={method!r} is unsupported for 1:N shortlist retrieval because "
        f"the secure split store only has persisted direct retrieval vector specs for {supported}. "
        f"Use one of {supported} for retrieval and optionally rerank the shortlist with {rerank}."
    )


def _sanitize_prefix(raw: str) -> str:
    s = _PREFIX_RE.sub("_", str(raw or "").strip().lower())
    if s and not s.endswith("_"):
        s += "_"
    return s


def resolve_biometric_database_url(explicit_database_url: str | None = None) -> str:
    return str(
        explicit_database_url
        or os.getenv("IDENTIFICATION_TEST_BIOMETRIC_DATABASE_URL")
        or os.getenv("IDENTIFICATION_TEST_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or DEFAULT_DATABASE_URL
    )


def resolve_identity_database_url(
    explicit_identity_database_url: str | None = None,
    *,
    biometric_database_url: str | None = None,
) -> str:
    biometric_url = str(biometric_database_url or resolve_biometric_database_url())
    return str(
        explicit_identity_database_url
        or os.getenv("IDENTIFICATION_TEST_IDENTITY_DATABASE_URL")
        or os.getenv("IDENTITY_DATABASE_URL")
        or biometric_url
    )


def _load_postgres_base_deps():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "PostgreSQL backend requires psycopg. Install project dependencies from environment.yml."
        ) from exc
    return psycopg, dict_row


def _load_pgvector_register():
    try:
        from pgvector.psycopg import register_vector
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "PostgreSQL backend requires pgvector Python bindings. Install project dependencies from environment.yml."
        ) from exc
    return register_vector


def _looks_like_missing_vector_type_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    return "vector type not found" in message or "type \"vector\" does not exist" in message


def _looks_like_missing_generic_vector_storage_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    return (
        "does not exist" in message
        or "undefined column" in message
        or "undefined table" in message
    )


class SecureSplitFingerprintStore:
    """
    PostgreSQL-backed secure split store for the 1:N identification stage.

    Recommended production layout:
      1. biometric_db
         - person_directory
         - raw_fingerprints
         - feature_vectors
      2. identity_db
         - identity_map

    Backward-compatible fallback:
      - If IDENTITY_DATABASE_URL is omitted, the store falls back to the same
        database URL for both sides and behaves like the previous single-DB
        logical split.

    Automatic bootstrap / migration:
      - If identity_db is empty and biometric_db still contains the legacy
        identity_map table, rows are copied automatically into identity_db.
      - If biometric_db is even older and still keeps national_id directly in
        person_directory, the mappings are copied automatically into identity_db
        (or into the same database in compatibility mode) and the legacy column
        is removed.
      - Duplicate or conflicting national_id mappings abort startup with a clear
        RuntimeError before any destructive cleanup happens.

    Transaction policy in dual-DB mode:
      - No two-phase commit.
      - Best effort with compensating rollback.
      - Enrollment commits identity_db first and biometric_db second, so a
        failed identity write can still rollback biometric work cleanly.
      - Purge deletes identity_db first only after biometric deletion is staged,
        then commits biometric_db; if the second commit fails, identity_db is
        restored from a snapshot.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        identity_database_url: str | None = None,
        table_prefix: str = "",
    ):
        self._configure_runtime_layout(
            database_url=database_url,
            identity_database_url=identity_database_url,
            table_prefix=table_prefix,
        )
        self._init_all()

    @classmethod
    def for_inspection(
        cls,
        database_url: str | None = None,
        *,
        identity_database_url: str | None = None,
        table_prefix: str = "",
    ) -> "SecureSplitFingerprintStore":
        store = cls.__new__(cls)
        store._configure_runtime_layout(
            database_url=database_url,
            identity_database_url=identity_database_url,
            table_prefix=table_prefix,
        )
        return store

    @classmethod
    def inspect_runtime_state(
        cls,
        database_url: str | None = None,
        *,
        identity_database_url: str | None = None,
        table_prefix: str = "",
    ) -> Dict[str, object]:
        return cls.for_inspection(
            database_url=database_url,
            identity_database_url=identity_database_url,
            table_prefix=table_prefix,
        ).collect_inspection_state()

    @classmethod
    def reconcile_runtime_state(
        cls,
        database_url: str | None = None,
        *,
        identity_database_url: str | None = None,
        table_prefix: str = "",
        repair_actions: Sequence[str] | None = None,
    ) -> Dict[str, object]:
        return cls.for_inspection(
            database_url=database_url,
            identity_database_url=identity_database_url,
            table_prefix=table_prefix,
        ).collect_reconciliation_report(repair_actions=repair_actions)

    def _configure_runtime_layout(
        self,
        *,
        database_url: str | None,
        identity_database_url: str | None,
        table_prefix: str,
    ) -> None:
        self.biometric_database_url = resolve_biometric_database_url(database_url)
        self.identity_database_url = resolve_identity_database_url(
            identity_database_url,
            biometric_database_url=self.biometric_database_url,
        )
        # Backward-compatible alias used by some helper scripts.
        self.database_url = self.biometric_database_url
        self.dual_database_enabled = self.identity_database_url != self.biometric_database_url

        self.table_prefix = _sanitize_prefix(table_prefix)

        self.person_table = f"{self.table_prefix}person_directory"
        self.identity_table = f"{self.table_prefix}identity_map"
        self.raw_table = f"{self.table_prefix}raw_fingerprints"
        self.vector_table = f"{self.table_prefix}feature_vectors"
        self.generic_vector_table = f"{self.table_prefix}method_retrieval_vectors"

        self.idx_person_created_at = f"{self.table_prefix}idx_person_created_at_desc"
        self.idx_person_name = f"{self.table_prefix}idx_person_name_norm_prefix"
        self.idx_identity_name = f"{self.table_prefix}idx_identity_name_norm_prefix"
        self.idx_identity_national = f"{self.table_prefix}idx_identity_national_id_prefix"
        self.legacy_idx_person_national = f"{self.table_prefix}idx_person_national_id_prefix"
        self.idx_vector_method_created_at = f"{self.table_prefix}idx_feature_vectors_method_created_at_desc"
        self.idx_generic_vector_method_kind_created_at = (
            f"{self.table_prefix}idx_method_retrieval_vectors_method_kind_created_at_desc"
        )
        self.idx_generic_vector_hnsw = {
            dim: f"{self.table_prefix}idx_method_retrieval_vectors_{dim}_hnsw"
            for dim in sorted(GENERIC_VECTOR_COLUMN_BY_DIM)
        }
        self.idx_vector_hnsw = {
            method: f"{self.table_prefix}idx_feature_vectors_{method}_hnsw"
            for method in sorted(_legacy_storage_methods())
        }
        self.idx_vector_dl = self.idx_vector_hnsw.get("dl", f"{self.table_prefix}idx_feature_vectors_dl_hnsw")
        self.idx_vector_vit = self.idx_vector_hnsw.get("vit", f"{self.table_prefix}idx_feature_vectors_vit_hnsw")
        self.ck_identity_full_name_not_blank = f"{self.table_prefix}ck_identity_full_name_not_blank"
        self.ck_identity_name_norm_not_blank = f"{self.table_prefix}ck_identity_name_norm_not_blank"
        self.ck_identity_name_norm_matches_full_name = f"{self.table_prefix}ck_identity_name_norm_matches_full_name"
        self.ck_identity_national_id_digits_only = f"{self.table_prefix}ck_identity_national_id_digits_only"

        self.layout_version = (
            "v4_dual_database_identity_profile_split"
            if self.dual_database_enabled
            else "v4_single_database_identity_profile_split_compat"
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def enroll(
        self,
        *,
        full_name: str,
        national_id: str,
        capture: str,
        ext: str,
        vectors: Dict[str, np.ndarray],
        image_bytes: bytes | None = None,
        image_sha256: str | None = None,
        byte_size: int | None = None,
        random_id: str | None = None,
        created_at: str | None = None,
        replace_existing: bool = False,
    ) -> EnrollmentReceipt:
        if not full_name or not str(full_name).strip():
            raise ValueError("full_name must not be empty")

        national_id_norm = normalize_national_id(national_id)
        if not national_id_norm:
            raise ValueError("national_id must contain at least one digit")

        capture_norm = str(capture or "plain").strip().lower() or "plain"
        if not ext:
            ext = ".png"
        if not ext.startswith("."):
            ext = f".{ext}"

        created_at_dt = self._coerce_timestamp(created_at)
        created_at_iso = created_at_dt.isoformat()
        random_id = random_id or uuid.uuid4().hex
        name_norm = normalize_name(full_name)
        if image_sha256 is None:
            if image_bytes is None:
                raise ValueError("image_bytes or image_sha256 must be provided")
            image_hash = sha256_bytes(image_bytes)
        else:
            image_hash = str(image_sha256).strip().lower()
            if not image_hash:
                raise ValueError("image_sha256 must not be empty")
        byte_size_value = byte_size
        if byte_size_value is None and image_bytes is not None:
            byte_size_value = len(image_bytes)
        if byte_size_value is not None:
            byte_size_value = int(byte_size_value)
            if byte_size_value < 0:
                raise ValueError("byte_size must be non-negative")

        vector_payload: Dict[str, np.ndarray] = {}
        for method, vec in vectors.items():
            method_norm = str(method).strip().lower()
            vector_payload[method_norm] = self._prepare_vector(method_norm, vec)
        if not vector_payload:
            raise ValueError("vectors must not be empty")

        if self.dual_database_enabled:
            return self._enroll_dual_database(
                full_name=full_name.strip(),
                name_norm=name_norm,
                national_id_norm=national_id_norm,
                capture_norm=capture_norm,
                ext=ext,
                vector_payload=vector_payload,
                random_id=random_id,
                created_at_dt=created_at_dt,
                created_at_iso=created_at_iso,
                image_hash=image_hash,
                byte_size=byte_size_value,
                replace_existing=replace_existing,
            )
        return self._enroll_single_database(
            full_name=full_name.strip(),
            name_norm=name_norm,
            national_id_norm=national_id_norm,
            capture_norm=capture_norm,
            ext=ext,
            vector_payload=vector_payload,
            random_id=random_id,
            created_at_dt=created_at_dt,
            created_at_iso=created_at_iso,
            image_hash=image_hash,
            byte_size=byte_size_value,
            replace_existing=replace_existing,
        )

    def purge(self, random_id: str) -> bool:
        if self.dual_database_enabled:
            return self._purge_dual_database(random_id)
        return self._purge_single_database(random_id)

    def total_people(self) -> int:
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) AS n FROM {self.person_table}")
                row = cur.fetchone()
        return int(row["n"]) if row else 0

    def count_vectors(self, method: str) -> int:
        method_norm = str(method).strip().lower()
        spec = self._spec_for_method(method_norm)
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                if spec.generic_storage_enabled and self._table_exists(cur, self.generic_vector_table):
                    if self._generic_vector_schema_compatible(cur):
                        generic_count = self._count_generic_vector_rows_for_spec(cur, spec=spec)
                        if generic_count > 0:
                            if not spec.legacy_storage_enabled:
                                return generic_count
                            legacy_count = self._count_legacy_vector_rows_for_spec(cur, spec=spec)
                            if generic_count >= legacy_count:
                                return generic_count
                            return legacy_count
                if not spec.legacy_storage_enabled:
                    return 0
                return self._count_legacy_vector_rows_for_spec(cur, spec=spec)

    def search_people(self, hints: IdentifyHints, *, limit: int | None = None) -> List[PersonDirectoryRecord]:
        if not self.dual_database_enabled:
            return self._search_people_single_database(hints, limit=limit)

        identity_map: Dict[str, _IdentityMapRow] = {}
        candidate_ids: Optional[List[str]] = None
        if hints.name_pattern or hints.national_id_pattern:
            identity_rows = self._query_identity_rows(
                name_pattern=hints.name_pattern,
                national_id_pattern=hints.national_id_pattern,
                candidate_ids=None,
                limit=None,
            )
            if not identity_rows:
                return []
            candidate_ids = [row.random_id for row in identity_rows]
            identity_map = {row.random_id: row for row in identity_rows}

        people_rows = self._query_people_rows(
            created_from=hints.created_from,
            created_to=hints.created_to,
            candidate_ids=candidate_ids,
            limit=limit,
        )
        if not people_rows:
            return []

        result_ids = [str(row['random_id']) for row in people_rows]
        missing_ids = [rid for rid in result_ids if rid not in identity_map]
        if missing_ids:
            for row in self._query_identity_rows(
                name_pattern=None,
                national_id_pattern=None,
                candidate_ids=missing_ids,
                limit=None,
            ):
                identity_map[row.random_id] = row

        records: List[PersonDirectoryRecord] = []
        for row in people_rows:
            random_id = str(row['random_id'])
            identity_row = identity_map.get(random_id)
            records.append(
                PersonDirectoryRecord(
                    random_id=random_id,
                    full_name='' if identity_row is None else identity_row.full_name,
                    name_norm='' if identity_row is None else identity_row.name_norm,
                    national_id='' if identity_row is None else identity_row.national_id,
                    created_at=self._to_iso(row['created_at']),
                )
            )
        return records
    def get_person(self, random_id: str) -> Optional[PersonDirectoryRecord]:
        if not self.dual_database_enabled:
            return self._get_person_single_database(random_id)

        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT random_id, created_at
                    FROM {self.person_table}
                    WHERE random_id = %s
                    """,
                    (random_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None

        identity_row = self._lookup_identity_row_by_random_id(random_id)
        return PersonDirectoryRecord(
            random_id=str(row['random_id']),
            full_name='' if identity_row is None else identity_row.full_name,
            name_norm='' if identity_row is None else identity_row.name_norm,
            national_id='' if identity_row is None else identity_row.national_id,
            created_at=self._to_iso(row['created_at']),
        )
    def list_random_ids_for_method(self, method: str) -> List[str]:
        method_norm = str(method).strip().lower()
        spec = self._spec_for_method(method_norm)
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                if spec.generic_storage_enabled and self._table_exists(cur, self.generic_vector_table):
                    if self._generic_vector_schema_compatible(cur):
                        generic_count = self._count_generic_vector_rows_for_spec(cur, spec=spec)
                        if generic_count > 0:
                            legacy_count = (
                                0
                                if not spec.legacy_storage_enabled
                                else self._count_legacy_vector_rows_for_spec(cur, spec=spec)
                            )
                            if spec.legacy_storage_enabled and generic_count < legacy_count:
                                cur.execute(
                                    f"SELECT random_id FROM {self.vector_table} "
                                    "WHERE method = %s ORDER BY created_at DESC",
                                    (spec.method,),
                                )
                                rows = cur.fetchall()
                                return [str(row["random_id"]) for row in rows]
                            cur.execute(
                                f"""
                                SELECT random_id
                                FROM {self.generic_vector_table}
                                WHERE method = %s
                                  AND vector_kind = %s
                                  AND dim = %s
                                  AND distance_metric = %s
                                ORDER BY created_at DESC
                                """,
                                (spec.method, spec.vector_kind, spec.dim, spec.distance_metric),
                            )
                            rows = cur.fetchall()
                            return [str(row["random_id"]) for row in rows]
                if not spec.legacy_storage_enabled:
                    return []
                cur.execute(
                    f"SELECT random_id FROM {self.vector_table} WHERE method = %s ORDER BY created_at DESC",
                    (spec.method,),
                )
                rows = cur.fetchall()
        return [str(row["random_id"]) for row in rows]

    def load_raw_fingerprint(self, random_id: str) -> Optional[RawFingerprintRecord]:
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        random_id,
                        capture,
                        ext,
                        sha256,
                        byte_size,
                        created_at,
                        image_bytes IS NOT NULL AS legacy_image_bytes_present
                    FROM {self.raw_table}
                    WHERE random_id = %s
                    """,
                    (random_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return RawFingerprintRecord(
            random_id=str(row["random_id"]),
            capture=str(row["capture"]),
            ext=str(row["ext"]),
            sha256=str(row["sha256"]),
            byte_size=None if row["byte_size"] is None else int(row["byte_size"]),
            created_at=self._to_iso(row["created_at"]),
            legacy_image_bytes_present=bool(row["legacy_image_bytes_present"]),
        )

    def load_legacy_raw_fingerprint_image_bytes(self, random_id: str) -> Optional[bytes]:
        """Return legacy persisted image bytes only for rows created before metadata-only enrollment."""
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT image_bytes
                    FROM {self.raw_table}
                    WHERE random_id = %s
                      AND image_bytes IS NOT NULL
                    """,
                    (random_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return bytes(row["image_bytes"])

    def load_vector(self, random_id: str, method: str) -> Optional[FeatureVectorRecord]:
        method_norm = str(method).strip().lower()
        spec = self._spec_for_method(method_norm)
        if spec.generic_storage_enabled:
            try:
                generic_row = self._load_generic_vector_row(random_id=random_id, method=method_norm)
            except Exception as exc:
                if not _looks_like_missing_generic_vector_storage_error(exc):
                    raise
                generic_row = None
            if generic_row is not None:
                return self._row_to_vector_record(generic_row)
        if not spec.legacy_storage_enabled:
            return None
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT random_id, method, dim, created_at, vector_512, vector_768
                    FROM {self.vector_table}
                    WHERE random_id = %s AND method = %s
                    """,
                    (random_id, method_norm),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_vector_record(row)

    def iter_vectors(self, random_ids: Sequence[str], method: str) -> Iterator[FeatureVectorRecord]:
        if not random_ids:
            return iter(())

        method_norm = str(method).strip().lower()
        spec = self._spec_for_method(method_norm)
        if spec.generic_storage_enabled:
            try:
                generic_rows = self._load_generic_vector_rows(random_ids=random_ids, method=method_norm)
            except Exception as exc:
                if not _looks_like_missing_generic_vector_storage_error(exc):
                    raise
                generic_rows = None
            if generic_rows is not None:
                return iter(self._row_to_vector_record(row) for row in generic_rows)
        if not spec.legacy_storage_enabled:
            return iter(())
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT random_id, method, dim, created_at, vector_512, vector_768
                    FROM {self.vector_table}
                    WHERE method = %s AND random_id = ANY(%s)
                    ORDER BY created_at DESC
                    """,
                    (spec.method, list(random_ids)),
                )
                rows = cur.fetchall()
        return iter(self._row_to_vector_record(row) for row in rows)

    def shortlist_by_vector(
        self,
        *,
        method: str,
        probe_vector: np.ndarray,
        limit: int,
        candidate_ids: Sequence[str] | None = None,
    ) -> List[tuple[str, float]]:
        method_norm = str(method).strip().lower()
        spec = self._spec_for_method(method_norm)
        probe = self._prepare_vector(method_norm, probe_vector)

        if spec.generic_storage_enabled:
            try:
                generic_result = self._shortlist_by_generic_vector(
                    spec=spec,
                    probe=probe,
                    limit=limit,
                    candidate_ids=candidate_ids,
                )
            except Exception as exc:
                if not _looks_like_missing_generic_vector_storage_error(exc):
                    raise
                generic_result = None
            if generic_result is not None:
                return generic_result

        if not spec.legacy_storage_enabled:
            return []

        return self._shortlist_by_legacy_vector(
            spec=spec,
            probe=probe,
            limit=limit,
            candidate_ids=candidate_ids,
        )

    def _shortlist_by_legacy_vector(
        self,
        *,
        spec: VectorSpec,
        probe: np.ndarray,
        limit: int,
        candidate_ids: Sequence[str] | None = None,
    ) -> List[tuple[str, float]]:
        if not spec.legacy_storage_enabled or spec.legacy_storage_column is None:
            return []
        method_norm = spec.method
        column = spec.legacy_storage_column

        sql_parts = [
            f"SELECT random_id, 1 - ({column} <=> %s::vector) AS retrieval_score",
            f"FROM {self.vector_table}",
            "WHERE method = %s",
        ]
        params: List[object] = [probe.tolist(), method_norm]

        if candidate_ids is not None:
            if not candidate_ids:
                return []
            sql_parts.append("AND random_id = ANY(%s)")
            params.append(list(candidate_ids))

        sql_parts.append(f"ORDER BY {column} <=> %s::vector, created_at DESC")
        sql_parts.append("LIMIT %s")
        params.append(probe.tolist())
        params.append(int(limit))

        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(" ".join(sql_parts), params)
                rows = cur.fetchall()
        return [(str(row["random_id"]), float(row["retrieval_score"])) for row in rows]

    def _shortlist_by_generic_vector(
        self,
        *,
        spec: VectorSpec,
        probe: np.ndarray,
        limit: int,
        candidate_ids: Sequence[str] | None = None,
    ) -> Optional[List[tuple[str, float]]]:
        if not self._should_use_generic_vector_storage(spec=spec, candidate_ids=candidate_ids):
            return None
        column = self._generic_vector_column_for_spec(spec)
        sql_parts = [
            f"SELECT random_id, 1 - ({column} <=> %s::vector) AS retrieval_score",
            f"FROM {self.generic_vector_table}",
            "WHERE method = %s",
            "AND vector_kind = %s",
            "AND dim = %s",
            "AND distance_metric = %s",
        ]
        params: List[object] = [probe.tolist(), spec.method, spec.vector_kind, spec.dim, spec.distance_metric]

        if candidate_ids is not None:
            if not candidate_ids:
                return []
            sql_parts.append("AND random_id = ANY(%s)")
            params.append(list(candidate_ids))

        sql_parts.append(f"ORDER BY {column} <=> %s::vector, created_at DESC")
        sql_parts.append("LIMIT %s")
        params.append(probe.tolist())
        params.append(int(limit))

        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(" ".join(sql_parts), params)
                rows = cur.fetchall()
        return [(str(row["random_id"]), float(row["retrieval_score"])) for row in rows]

    def _load_generic_vector_row(self, *, random_id: str, method: str):
        spec = self._spec_for_method(method)
        if not spec.generic_storage_enabled:
            return None
        _ = self._generic_vector_column_for_spec(spec)
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                if not self._table_exists(cur, self.generic_vector_table):
                    return None
                if not self._generic_vector_schema_compatible(cur):
                    return None
                cur.execute(
                    f"""
                    SELECT random_id, method, dim, created_at, vector_512, vector_768
                    FROM {self.generic_vector_table}
                    WHERE random_id = %s
                      AND method = %s
                      AND vector_kind = %s
                      AND dim = %s
                      AND distance_metric = %s
                    """,
                    (random_id, spec.method, spec.vector_kind, spec.dim, spec.distance_metric),
                )
                return cur.fetchone()

    def _load_generic_vector_rows(self, *, random_ids: Sequence[str], method: str):
        spec = self._spec_for_method(method)
        if not spec.generic_storage_enabled:
            return None
        _ = self._generic_vector_column_for_spec(spec)
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                if not self._table_exists(cur, self.generic_vector_table):
                    return None
                if not self._generic_vector_schema_compatible(cur):
                    return None
                generic_count = self._count_generic_vector_rows_for_spec(
                    cur,
                    spec=spec,
                    candidate_ids=random_ids,
                )
                legacy_count = self._count_legacy_vector_rows_for_spec(
                    cur,
                    spec=spec,
                    candidate_ids=random_ids,
                )
                if legacy_count > generic_count:
                    return None
                if generic_count == 0:
                    return []
                cur.execute(
                    f"""
                    SELECT random_id, method, dim, created_at, vector_512, vector_768
                    FROM {self.generic_vector_table}
                    WHERE method = %s
                      AND vector_kind = %s
                      AND dim = %s
                      AND distance_metric = %s
                      AND random_id = ANY(%s)
                    ORDER BY created_at DESC
                    """,
                    (spec.method, spec.vector_kind, spec.dim, spec.distance_metric, list(random_ids)),
                )
                return cur.fetchall()

    def _should_use_generic_vector_storage(
        self,
        *,
        spec: VectorSpec,
        candidate_ids: Sequence[str] | None = None,
    ) -> bool:
        if not spec.generic_storage_enabled:
            return False
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                if not self._table_exists(cur, self.generic_vector_table):
                    return False
                if not self._generic_vector_schema_compatible(cur):
                    return False
                generic_count = self._count_generic_vector_rows_for_spec(
                    cur,
                    spec=spec,
                    candidate_ids=candidate_ids,
                )
                if generic_count <= 0:
                    return False
                if not spec.legacy_storage_enabled:
                    return True
                legacy_count = self._count_legacy_vector_rows_for_spec(
                    cur,
                    spec=spec,
                    candidate_ids=candidate_ids,
                )
                return generic_count >= legacy_count

    def _count_generic_vector_rows_for_spec(
        self,
        cur,
        *,
        spec: VectorSpec,
        candidate_ids: Sequence[str] | None = None,
    ) -> int:
        if not spec.generic_storage_enabled:
            return 0
        clauses = [
            "method = %s",
            "vector_kind = %s",
            "dim = %s",
            "distance_metric = %s",
        ]
        params: List[object] = [spec.method, spec.vector_kind, spec.dim, spec.distance_metric]
        if candidate_ids is not None:
            if not candidate_ids:
                return 0
            clauses.append("random_id = ANY(%s)")
            params.append(list(candidate_ids))
        cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {self.generic_vector_table}
            WHERE {" AND ".join(clauses)}
            """,
            params,
        )
        row = cur.fetchone()
        return 0 if row is None else int(row["n"])

    def _count_legacy_vector_rows_for_spec(
        self,
        cur,
        *,
        spec: VectorSpec,
        candidate_ids: Sequence[str] | None = None,
    ) -> int:
        if not spec.legacy_storage_enabled:
            return 0
        if not self._table_exists(cur, self.vector_table):
            return 0
        clauses = ["method = %s"]
        params: List[object] = [spec.method]
        if candidate_ids is not None:
            if not candidate_ids:
                return 0
            clauses.append("random_id = ANY(%s)")
            params.append(list(candidate_ids))
        cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {self.vector_table}
            WHERE {" AND ".join(clauses)}
            """,
            params,
        )
        row = cur.fetchone()
        return 0 if row is None else int(row["n"])

    @staticmethod
    def _generic_vector_column_for_spec(spec: VectorSpec) -> str:
        if not spec.generic_storage_enabled:
            raise ValueError(f"retrieval method {spec.method!r} does not enable generic vector storage")
        if spec.distance_metric != "cosine":
            raise ValueError(
                f"retrieval method {spec.method!r} declares distance_metric={spec.distance_metric!r}, "
                "but generic indexed retrieval storage currently supports cosine pgvector search only"
            )
        column = spec.generic_storage_column or GENERIC_VECTOR_COLUMN_BY_DIM.get(int(spec.dim))
        if column is None or column not in GENERIC_VECTOR_COLUMN_BY_DIM.values():
            raise ValueError(
                f"retrieval method {spec.method!r} declares vector dim {spec.dim}, but generic retrieval "
                "storage has no indexed pgvector column for that dimension"
            )
        return column

    def dump_layout(self) -> Dict[str, str]:
        biometric_name = self._database_name_from_url(self.biometric_database_url)
        identity_name = self._database_name_from_url(self.identity_database_url)
        return {
            "backend": "postgresql",
            "layout_version": self.layout_version,
            "dual_database_enabled": "true" if self.dual_database_enabled else "false",
            "biometric_database_url": self._redacted_database_url(self.biometric_database_url),
            "identity_database_url": self._redacted_database_url(self.identity_database_url),
            "person_table": f"{biometric_name}.{self.person_table}",
            "raw_fingerprints_table": f"{biometric_name}.{self.raw_table}",
            "feature_vectors_table": f"{biometric_name}.{self.vector_table}",
            "method_retrieval_vectors_table": f"{biometric_name}.{self.generic_vector_table}",
            "identity_map_table": f"{identity_name}.{self.identity_table}",
            "vector_storage_mode": str(VECTOR_STORAGE_SCHEMA["mode"]),
            "method_generic_vectors_supported": "true",
        }

    @staticmethod
    def method_capability_metadata() -> Dict[str, Dict[str, object]]:
        return load_api_method_registry().method_capabilities()

    @staticmethod
    def vector_storage_schema_metadata() -> Dict[str, object]:
        return {
            **VECTOR_STORAGE_SCHEMA,
            "legacy_compatibility_methods": sorted(_legacy_storage_methods()),
            "generic_storage_methods": sorted(_generic_storage_methods()),
            "generic_only_methods": sorted(_generic_only_methods()),
            "dual_write_methods": sorted(_dual_write_methods()),
            "configured_vector_specs": {
                method: {
                    "method": spec.method,
                    "vector_dim": spec.dim,
                    "vector_kind": spec.vector_kind,
                    "distance_metric": spec.distance_metric,
                    "legacy_storage_enabled": spec.legacy_storage_enabled,
                    "generic_storage_enabled": spec.generic_storage_enabled,
                    "preferred_storage": spec.preferred_storage,
                    "legacy_storage_column": (
                        spec.legacy_storage_column if spec.legacy_storage_enabled else None
                    ),
                    "generic_storage_column": spec.generic_storage_column,
                    "storage_column": spec.generic_storage_column,
                }
                for method, spec in sorted(VECTOR_SPECS.items())
            },
            "schema_accepts_method_generic_vectors": True,
            "schema_check_constraint_mode": "generic_dim_vector_column_without_method_names",
            "generic_table": "method_retrieval_vectors",
            "generic_primary_key": ["random_id", "method", "vector_kind"],
        }

    def collect_reconciliation_report(
        self,
        *,
        repair_actions: Sequence[str] | None = None,
    ) -> Dict[str, object]:
        requested_repairs = self._normalize_repair_actions(repair_actions)
        initial_inspection = self.collect_inspection_state()
        available_repairs = self._available_repair_actions(initial_inspection["issues"])
        applied_repairs: List[Dict[str, object]] = []

        current_inspection = initial_inspection
        if requested_repairs:
            applied_repairs = self._apply_requested_repairs(requested_repairs)
            current_inspection = self.collect_inspection_state()

        summary = self._build_reconciliation_summary(current_inspection["issues"])
        summary["overall_ok"] = bool(current_inspection["overall_ok"])
        summary["readiness"] = dict(current_inspection.get("readiness", {}))
        report: Dict[str, object] = {
            "generated_at": utc_now_iso(),
            "report_mode": "repair" if requested_repairs else "report_only",
            "requested_repairs": requested_repairs,
            "available_repairs": available_repairs,
            "applied_repairs": applied_repairs,
            "summary": summary,
            "commands": self._reconciliation_command_examples(),
            "inspection": current_inspection,
            "issues": list(current_inspection["issues"]),
        }
        if requested_repairs:
            report["inspection_before_repairs"] = initial_inspection
        return report

    def backfill_generic_retrieval_vectors(self) -> Dict[str, object]:
        """Copy legacy dl/vit pgvector rows into method-generic retrieval storage."""
        return self._backfill_generic_retrieval_vectors()

    def collect_inspection_state(self) -> Dict[str, object]:
        biometric_name = self._database_name_from_url(self.biometric_database_url)
        identity_name = self._database_name_from_url(self.identity_database_url)
        issues: List[Dict[str, object]] = []

        biometric_snapshot = self._inspect_database_role(
            database_url=self.biometric_database_url,
            database_role="biometric_db",
            include_vector_extension=True,
            issues=issues,
        )
        if self.dual_database_enabled:
            identity_snapshot = self._inspect_database_role(
                database_url=self.identity_database_url,
                database_role="identity_db",
                include_vector_extension=False,
                issues=issues,
            )
        else:
            identity_snapshot = biometric_snapshot

        self._append_expected_table_issues(
            issues,
            database_role="biometric_db",
            snapshot=biometric_snapshot,
            expected={"person", "raw", "vectors"},
        )
        self._append_expected_table_issues(
            issues,
            database_role="identity_db",
            snapshot=identity_snapshot,
            expected={"identity"},
        )
        self._append_legacy_layout_issues(issues, biometric_snapshot, identity_snapshot)
        self._append_vector_storage_migration_issues(issues, biometric_snapshot)
        reconciliation = self._append_integrity_issues(issues, biometric_snapshot, identity_snapshot)
        schema_hardening = self._build_schema_hardening_state(
            biometric_snapshot=biometric_snapshot,
            identity_snapshot=identity_snapshot,
        )
        template_protection = self._build_template_protection_state(
            biometric_snapshot=biometric_snapshot,
        )
        self._append_template_protection_issues(
            issues,
            template_protection=template_protection,
        )
        self._append_schema_hardening_issues(
            issues,
            biometric_snapshot=biometric_snapshot,
            identity_snapshot=identity_snapshot,
            schema_hardening=schema_hardening,
        )
        reconciliation["repairability_summary"] = self._summarize_issue_repairability(issues)
        errors = [dict(issue) for issue in issues if str(issue.get("severity")) == "error"]
        warnings = [dict(issue) for issue in issues if str(issue.get("severity")) != "error"]
        overall_ok = not errors
        readiness_status = "not_ready"
        if overall_ok:
            readiness_status = "ready_with_warnings" if warnings else "ready"
        registry = load_api_method_registry()
        method_capabilities = registry.method_capabilities()
        retrieval_vector_coverage = self._build_retrieval_vector_coverage_state(
            biometric_snapshot=biometric_snapshot,
            registry=registry,
        )

        return {
            "backend": "postgresql",
            "layout_version": self.layout_version,
            "dual_database_enabled": self.dual_database_enabled,
            "table_prefix": self.table_prefix,
            "redacted_database_urls": {
                "biometric_db": self._redacted_database_url(self.biometric_database_url),
                "identity_db": self._redacted_database_url(self.identity_database_url),
            },
            "resolved_table_names": {
                "person": f"{biometric_name}.{self.person_table}",
                "identity": f"{identity_name}.{self.identity_table}",
                "raw": f"{biometric_name}.{self.raw_table}",
                "vectors": f"{biometric_name}.{self.vector_table}",
                "generic_vectors": f"{biometric_name}.{self.generic_vector_table}",
            },
            "table_presence": {
                "biometric_db": dict(biometric_snapshot["table_presence"]),
                "identity_db": dict(identity_snapshot["table_presence"]),
            },
            "vector_extension_present_in_biometric_db": biometric_snapshot["vector_extension_present"],
            "row_counts": {
                "people": biometric_snapshot["row_counts"]["people"],
                "identity": identity_snapshot["row_counts"]["identity"],
                "raw": biometric_snapshot["row_counts"]["raw"],
                "vectors_by_method": dict(biometric_snapshot["row_counts"]["vectors_by_method"]),
                "legacy_vectors_by_method": dict(biometric_snapshot["row_counts"]["vectors_by_method"]),
                "generic_vectors_by_method_kind": dict(
                    biometric_snapshot["row_counts"]["generic_vectors_by_method_kind"]
                ),
            },
            "unexpected_vector_methods": dict(biometric_snapshot["row_counts"]["unexpected_vector_methods"]),
            "method_capabilities": method_capabilities,
            "retrieval_capabilities": method_capabilities,
            "direct_vector_retrieval_methods": list(registry.direct_vector_retrieval_methods()),
            "rerank_only_methods": list(registry.rerank_only_methods()),
            "retrieval_vector_coverage_by_method": retrieval_vector_coverage["coverage_by_method"],
            "retrieval_methods_missing_vectors": retrieval_vector_coverage["methods_missing_vectors"],
            "retrieval_methods_with_zero_coverage": retrieval_vector_coverage["methods_with_zero_coverage"],
            "coverage_recommendation": retrieval_vector_coverage["recommendation"],
            "vector_storage_schema": self._build_vector_storage_schema_state(biometric_snapshot),
            "schema_hardening": schema_hardening,
            "template_protection": template_protection,
            "reconciliation": reconciliation,
            "integrity_warnings": [str(issue["message"]) for issue in warnings],
            "overall_ok": overall_ok,
            "readiness": {
                "ready": overall_ok,
                "status": readiness_status,
                "error_count": len(errors),
                "warning_count": len(warnings),
            },
            "errors": errors,
            "warnings": warnings,
            "issues": issues,
        }

    def _inspect_database_role(
        self,
        *,
        database_url: str,
        database_role: str,
        include_vector_extension: bool,
        issues: List[Dict[str, object]],
    ) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "database_role": database_role,
            "database_name": self._database_name_from_url(database_url),
            "redacted_database_url": self._redacted_database_url(database_url),
            "connection_ok": False,
            "table_presence": {
                "person": False,
                "identity": False,
                "raw": False,
                "vectors": False,
                "generic_vectors": False,
            },
            "row_counts": {
                "people": None,
                "identity": None,
                "raw": None,
                "vectors_by_method": {method: None for method in sorted(_legacy_storage_methods())},
                "generic_vectors_by_method_kind": {},
                "unexpected_vector_methods": {},
            },
            "generic_vector_schema_compatible": None,
            "vector_extension_present": None,
            "legacy_person_columns": {
                "national_id": False,
                "full_name": False,
                "name_norm": False,
            },
            "index_presence": {
                spec["key"]: False for spec in self._schema_index_contract()
            },
            "identity_constraint_presence": {
                spec["key"]: False for spec in self._identity_constraint_contract()
            },
            "identity_profile_missing_sample": [],
            "identity_profile_missing_count": 0,
            "raw_orphan_sample": [],
            "raw_orphan_count": 0,
            "vector_orphan_sample": [],
            "vector_orphan_count": 0,
            "raw_image_bytes_column_present": None,
            "raw_image_bytes_column_nullable": None,
            "legacy_raw_image_bytes_row_count": None,
            "legacy_raw_image_bytes_sample": [],
        }

        try:
            with self._connect_postgres_inspection(database_url, database_role=database_role) as conn:
                snapshot["connection_ok"] = True
                with conn.cursor() as cur:
                    table_presence = snapshot["table_presence"]
                    table_presence["person"] = self._table_exists(cur, self.person_table)
                    table_presence["identity"] = self._table_exists(cur, self.identity_table)
                    table_presence["raw"] = self._table_exists(cur, self.raw_table)
                    table_presence["vectors"] = self._table_exists(cur, self.vector_table)
                    table_presence["generic_vectors"] = self._table_exists(cur, self.generic_vector_table)
                    snapshot["index_presence"] = self._collect_schema_index_presence(
                        cur,
                        table_presence=table_presence,
                    )
                    if table_presence["identity"]:
                        snapshot["identity_constraint_presence"] = self._collect_identity_constraint_presence(cur)

                    if include_vector_extension:
                        snapshot["vector_extension_present"] = self._has_vector_extension(cur)

                    if table_presence["person"]:
                        snapshot["row_counts"]["people"] = self._count_rows(cur, self.person_table)
                        snapshot["legacy_person_columns"] = {
                            "national_id": self._column_exists(cur, self.person_table, "national_id"),
                            "full_name": self._column_exists(cur, self.person_table, "full_name"),
                            "name_norm": self._column_exists(cur, self.person_table, "name_norm"),
                        }

                    if table_presence["identity"]:
                        snapshot["row_counts"]["identity"] = self._count_rows(cur, self.identity_table)
                        snapshot["identity_profile_missing_sample"] = self._identity_rows_missing_profile(cur)
                        snapshot["identity_profile_missing_count"] = self._count_identity_rows_missing_profile(cur)

                    if table_presence["raw"]:
                        snapshot["row_counts"]["raw"] = self._count_rows(cur, self.raw_table)
                        raw_image_bytes_column_present = self._column_exists(cur, self.raw_table, "image_bytes")
                        snapshot["raw_image_bytes_column_present"] = raw_image_bytes_column_present
                        if raw_image_bytes_column_present:
                            snapshot["raw_image_bytes_column_nullable"] = not self._column_is_not_null(
                                cur,
                                self.raw_table,
                                "image_bytes",
                            )
                            snapshot["legacy_raw_image_bytes_row_count"] = (
                                self._count_legacy_raw_image_bytes_rows(cur)
                            )
                            snapshot["legacy_raw_image_bytes_sample"] = (
                                self._sample_legacy_raw_image_bytes_random_ids(cur)
                            )
                        else:
                            snapshot["raw_image_bytes_column_nullable"] = None
                            snapshot["legacy_raw_image_bytes_row_count"] = 0
                            snapshot["legacy_raw_image_bytes_sample"] = []
                        if table_presence["person"]:
                            snapshot["raw_orphan_count"] = self._count_orphan_rows(
                                cur,
                                child_table=self.raw_table,
                                parent_table=self.person_table,
                            )
                            snapshot["raw_orphan_sample"] = self._sample_orphan_ids(
                                cur,
                                child_table=self.raw_table,
                                parent_table=self.person_table,
                            )

                    if table_presence["vectors"]:
                        method_counts = self._count_vector_rows_by_method(cur)
                        for method in snapshot["row_counts"]["vectors_by_method"]:
                            snapshot["row_counts"]["vectors_by_method"][method] = int(method_counts.get(method, 0))
                        snapshot["row_counts"]["unexpected_vector_methods"] = {
                            method: int(count)
                            for method, count in method_counts.items()
                            if method not in _legacy_storage_methods()
                        }
                        if table_presence["person"]:
                            snapshot["vector_orphan_count"] = self._count_orphan_rows(
                                cur,
                                child_table=self.vector_table,
                                parent_table=self.person_table,
                            )
                            snapshot["vector_orphan_sample"] = self._sample_orphan_ids(
                                cur,
                                child_table=self.vector_table,
                                parent_table=self.person_table,
                            )
                    if table_presence["generic_vectors"]:
                        snapshot["generic_vector_schema_compatible"] = self._generic_vector_schema_compatible(cur)
                        snapshot["row_counts"]["generic_vectors_by_method_kind"] = (
                            self._count_generic_vector_rows_by_method_kind(cur)
                        )
        except Exception as exc:
            self._append_issue(
                issues,
                code="database_connection_failed",
                severity="error",
                database_role=database_role,
                message=str(exc),
            )

        return snapshot

    def _append_expected_table_issues(
        self,
        issues: List[Dict[str, object]],
        *,
        database_role: str,
        snapshot: Dict[str, Any],
        expected: set[str],
    ) -> None:
        if not snapshot["connection_ok"]:
            return
        for table_key in sorted(expected):
            if snapshot["table_presence"].get(table_key):
                continue
            self._append_issue(
                issues,
                code="missing_table",
                severity="error",
                database_role=database_role,
                message=f"{database_role} is missing expected table {self._logical_table_name(table_key)}.",
                table=self._logical_table_name(table_key),
            )

    def _append_legacy_layout_issues(
        self,
        issues: List[Dict[str, object]],
        biometric_snapshot: Dict[str, Any],
        identity_snapshot: Dict[str, Any],
    ) -> None:
        if biometric_snapshot["connection_ok"]:
            legacy_columns = biometric_snapshot["legacy_person_columns"]
            if legacy_columns["national_id"]:
                self._append_issue(
                    issues,
                    code="legacy_person_national_id_column_present",
                    severity="warning",
                    database_role="biometric_db",
                    message=f"{self.person_table}.national_id still exists in biometric_db.",
                    table=self.person_table,
                    column="national_id",
                )
            if legacy_columns["full_name"] or legacy_columns["name_norm"]:
                self._append_issue(
                    issues,
                    code="legacy_person_profile_columns_present",
                    severity="warning",
                    database_role="biometric_db",
                    message=f"{self.person_table} still contains legacy profile columns in biometric_db.",
                    table=self.person_table,
                )
            if self.dual_database_enabled and biometric_snapshot["table_presence"]["identity"]:
                self._append_issue(
                    issues,
                    code="legacy_identity_table_present_in_biometric_db",
                    severity="warning",
                    database_role="biometric_db",
                    message=f"Legacy table {self.identity_table} still exists in biometric_db.",
                    table=self.identity_table,
                )
            if biometric_snapshot["vector_extension_present"] is False:
                self._append_issue(
                    issues,
                    code="missing_vector_extension",
                    severity="error",
                    database_role="biometric_db",
                    message="pgvector extension is not installed in biometric_db.",
                )

        if identity_snapshot["connection_ok"] and identity_snapshot["identity_profile_missing_sample"]:
            self._append_issue(
                issues,
                code="identity_rows_missing_profile",
                severity="warning",
                database_role="identity_db",
                message=(
                    f"{self.identity_table} contains rows missing full_name/name_norm; sample random_id values: "
                    + ", ".join(identity_snapshot["identity_profile_missing_sample"])
                ),
                table=self.identity_table,
                row_count=int(identity_snapshot["identity_profile_missing_count"]),
                sample_random_ids=list(identity_snapshot["identity_profile_missing_sample"]),
            )

        unexpected_vector_methods = biometric_snapshot["row_counts"]["unexpected_vector_methods"]
        if unexpected_vector_methods:
            self._append_issue(
                issues,
                code="unexpected_vector_methods_present",
                severity="warning",
                database_role="biometric_db",
                message=(
                    "feature_vectors contains unexpected methods: "
                    + ", ".join(
                        f"{method}={count}"
                        for method, count in sorted(unexpected_vector_methods.items())
                    )
                ),
                table=self.vector_table,
                methods=dict(unexpected_vector_methods),
            )

    def _append_vector_storage_migration_issues(
        self,
        issues: List[Dict[str, object]],
        biometric_snapshot: Dict[str, Any],
    ) -> None:
        storage_state = self._build_vector_storage_schema_state(biometric_snapshot)
        if not bool(storage_state.get("backfill_recommended")):
            return
        missing = {
            method: details
            for method, details in dict(storage_state.get("generic_population_by_method", {})).items()
            if bool(dict(details).get("backfill_recommended"))
        }
        self._append_issue(
            issues,
            code="generic_vector_backfill_recommended",
            severity="warning",
            database_role="biometric_db",
            message=(
                f"{self.generic_vector_table} is not fully populated for legacy direct retrieval vectors; "
                "run the generic vector backfill before relying exclusively on method-generic storage."
            ),
            table=self.generic_vector_table,
            source_table=self.vector_table,
            methods=missing,
        )

    def _build_retrieval_vector_coverage_state(
        self,
        *,
        biometric_snapshot: Dict[str, Any],
        registry,
    ) -> Dict[str, object]:
        people_count_raw = biometric_snapshot["row_counts"].get("people")
        people_count = None if people_count_raw is None else int(people_count_raw)
        legacy_counts = {
            method: (None if count is None else int(count))
            for method, count in dict(biometric_snapshot["row_counts"]["vectors_by_method"]).items()
        }
        generic_counts = {
            str(key): int(value)
            for key, value in dict(biometric_snapshot["row_counts"].get("generic_vectors_by_method_kind", {})).items()
        }
        generic_table_present = bool(biometric_snapshot["table_presence"].get("generic_vectors"))

        coverage_by_method: Dict[str, Dict[str, object]] = {}
        methods_missing_vectors: List[str] = []
        methods_with_zero_coverage: List[str] = []
        for method in registry.direct_vector_retrieval_methods():
            spec = VECTOR_SPECS.get(method)
            if spec is None:
                continue

            legacy_count = legacy_counts.get(method) if spec.legacy_storage_enabled else None
            generic_key = self._generic_vector_count_key(method=method, vector_kind=spec.vector_kind)
            generic_count = (
                int(generic_counts.get(generic_key, 0))
                if generic_table_present and spec.generic_storage_enabled
                else None
            )
            available_counts = [
                count
                for count in (legacy_count, generic_count)
                if count is not None
            ]
            available_count = max(available_counts) if available_counts else 0
            missing_count = None if people_count is None else max(people_count - available_count, 0)
            coverage_ratio = (
                None
                if people_count is None or people_count <= 0
                else min(1.0, available_count / people_count)
            )

            if people_count is None:
                coverage_status = "unknown"
            elif people_count <= 0:
                coverage_status = "no_identities"
            elif available_count <= 0:
                coverage_status = "zero"
            elif missing_count and missing_count > 0:
                coverage_status = "partial"
            else:
                coverage_status = "full"

            if missing_count and missing_count > 0:
                methods_missing_vectors.append(method)
            if people_count is not None and people_count > 0 and available_count <= 0:
                methods_with_zero_coverage.append(method)

            coverage_by_method[method] = {
                "method": method,
                "vector_kind": spec.vector_kind,
                "expected_identity_count": people_count,
                "legacy_row_count": legacy_count,
                "generic_row_count": generic_count,
                "available_row_count": available_count,
                "missing_vector_count": missing_count,
                "coverage_ratio": coverage_ratio,
                "coverage_status": coverage_status,
                "preferred_storage": spec.preferred_storage,
                "generic_storage_enabled": spec.generic_storage_enabled,
                "legacy_storage_enabled": spec.legacy_storage_enabled,
                "source_path_required_for_reembedding": True,
            }

        if people_count is None:
            recommendation = (
                "Retrieval vector coverage could not be evaluated because the enrolled identity count "
                "was not available."
            )
        elif people_count <= 0:
            recommendation = (
                "No enrolled identities are present, so retrieval vector coverage will populate on the next "
                "enrollment or demo/browser seed."
            )
        elif methods_missing_vectors:
            recommendation = (
                "One or more direct retrieval methods have fewer vectors than enrolled identities. "
                "Reseed demo/browser stores or re-enroll affected operational identities with the missing "
                "methods. Existing operational rows cannot be generically re-embedded unless a trusted source "
                "image path is known, because new operational uploads persist metadata and vectors rather "
                "than raw image bytes."
            )
        else:
            recommendation = (
                "All configured direct-vector retrieval methods have vector coverage for the current "
                "enrolled identities."
            )

        return {
            "coverage_by_method": coverage_by_method,
            "methods_missing_vectors": methods_missing_vectors,
            "methods_with_zero_coverage": methods_with_zero_coverage,
            "recommendation": recommendation,
        }

    def _build_vector_storage_schema_state(self, biometric_snapshot: Dict[str, Any]) -> Dict[str, object]:
        legacy_counts = {
            method: (None if count is None else int(count))
            for method, count in dict(biometric_snapshot["row_counts"]["vectors_by_method"]).items()
        }
        generic_counts = {
            str(key): int(value)
            for key, value in dict(biometric_snapshot["row_counts"].get("generic_vectors_by_method_kind", {})).items()
        }
        generic_table_present = bool(biometric_snapshot["table_presence"].get("generic_vectors"))
        generic_schema_compatible = bool(biometric_snapshot.get("generic_vector_schema_compatible"))
        generic_population_by_method: Dict[str, Dict[str, object]] = {}
        backfill_recommended = False
        legacy_fallback_used = False
        for method, spec in sorted(VECTOR_SPECS.items()):
            legacy_count_raw = legacy_counts.get(method) if spec.legacy_storage_enabled else None
            legacy_count = 0 if legacy_count_raw is None else int(legacy_count_raw)
            generic_key = self._generic_vector_count_key(method=method, vector_kind=spec.vector_kind)
            generic_count = (
                int(generic_counts.get(generic_key, 0))
                if generic_table_present and spec.generic_storage_enabled
                else 0
            )
            method_backfill_recommended = (
                spec.legacy_storage_enabled
                and spec.generic_storage_enabled
                and legacy_count > generic_count
            )
            method_legacy_fallback = (
                spec.legacy_storage_enabled
                and method_backfill_recommended
                and legacy_count > 0
            )
            backfill_recommended = backfill_recommended or method_backfill_recommended
            legacy_fallback_used = legacy_fallback_used or method_legacy_fallback
            generic_population_by_method[method] = {
                "vector_kind": spec.vector_kind,
                "legacy_storage_enabled": spec.legacy_storage_enabled,
                "generic_storage_enabled": spec.generic_storage_enabled,
                "preferred_storage": spec.preferred_storage,
                "legacy_row_count": legacy_count_raw,
                "generic_row_count": generic_count if generic_table_present else None,
                "generic_populated": (
                    spec.generic_storage_enabled and generic_table_present and generic_count > 0
                ),
                "generic_covers_legacy": (
                    generic_table_present
                    and (
                        not spec.legacy_storage_enabled
                        or generic_count >= legacy_count
                    )
                ),
                "legacy_fallback_used": method_legacy_fallback,
                "backfill_recommended": method_backfill_recommended,
            }

        return {
            **self.vector_storage_schema_metadata(),
            "legacy_feature_vectors_table_present": bool(biometric_snapshot["table_presence"].get("vectors")),
            "generic_retrieval_vectors_table_present": generic_table_present,
            "generic_table_name": self.generic_vector_table,
            "generic_schema_compatible": generic_schema_compatible,
            "method_generic_vectors_supported": generic_table_present and generic_schema_compatible,
            "generic_rows_by_method_vector_kind": generic_counts,
            "legacy_rows_by_method": legacy_counts,
            "generic_population_by_method": generic_population_by_method,
            "legacy_fallback_used": legacy_fallback_used,
            "backfill_recommended": backfill_recommended,
            "backfill_action": "backfill_generic_retrieval_vectors",
            "backfill_command": self._reconciliation_script_command("--backfill-generic-retrieval-vectors"),
        }

    def _build_template_protection_state(
        self,
        *,
        biometric_snapshot: Dict[str, Any],
    ) -> Dict[str, object]:
        raw_table_present = bool(biometric_snapshot["table_presence"].get("raw"))
        column_present = biometric_snapshot.get("raw_image_bytes_column_present")
        column_nullable = biometric_snapshot.get("raw_image_bytes_column_nullable")
        legacy_count = biometric_snapshot.get("legacy_raw_image_bytes_row_count")
        legacy_sample = [
            str(random_id)
            for random_id in list(biometric_snapshot.get("legacy_raw_image_bytes_sample", []))
        ]

        status = "unknown"
        if biometric_snapshot["connection_ok"]:
            if not raw_table_present:
                column_present = False
                column_nullable = None
                legacy_count = None
                legacy_sample = []
            elif column_present is False:
                status = "no_legacy_column"
                legacy_count = 0
                legacy_sample = []
            elif column_present is True and legacy_count is not None:
                status = "legacy_payloads_present" if int(legacy_count) > 0 else "clear"

        return {
            "raw_image_storage_policy": "metadata_only_new_writes",
            "new_raw_image_persistence_enabled": False,
            "database_role": "biometric_db",
            "table": self.raw_table,
            "column": "image_bytes",
            "raw_table_present": raw_table_present if biometric_snapshot["connection_ok"] else None,
            "raw_image_bytes_column_present": column_present,
            "raw_image_bytes_column_nullable": column_nullable,
            "legacy_raw_image_bytes_row_count": legacy_count,
            "legacy_raw_image_bytes_sample": legacy_sample,
            "legacy_raw_image_storage_status": status,
            "rerank_legacy_bytes_adapter_enabled": True,
        }

    def _append_template_protection_issues(
        self,
        issues: List[Dict[str, object]],
        *,
        template_protection: Dict[str, object],
    ) -> None:
        if template_protection.get("legacy_raw_image_storage_status") != "legacy_payloads_present":
            return
        row_count = int(template_protection.get("legacy_raw_image_bytes_row_count") or 0)
        sample_random_ids = [
            str(random_id)
            for random_id in list(template_protection.get("legacy_raw_image_bytes_sample", []))
        ]
        self._append_issue(
            issues,
            code="legacy_raw_image_bytes_present",
            severity="warning",
            database_role="biometric_db",
            message=(
                f"{self.raw_table}.image_bytes still contains legacy raw biometric image bytes in "
                f"{row_count} row(s). Redact these payloads to satisfy template-protection and "
                "data-minimization policy while preserving metadata and vectors."
            ),
            table=self.raw_table,
            column="image_bytes",
            row_count=row_count,
            sample_random_ids=sample_random_ids,
        )

    def _append_integrity_issues(
        self,
        issues: List[Dict[str, object]],
        biometric_snapshot: Dict[str, Any],
        identity_snapshot: Dict[str, Any],
    ) -> Dict[str, object]:
        reconciliation: Dict[str, object] = {
            "validation_mode": (
                "bounded_exact_streaming" if self.dual_database_enabled else "bounded_sql_antijoin"
            ),
            "sample_limit": RECONCILIATION_SAMPLE_LIMIT,
            "sample_counts": {
                "raw_rows_without_person": len(biometric_snapshot["raw_orphan_sample"]),
                "vector_rows_without_person": len(biometric_snapshot["vector_orphan_sample"]),
                "people_without_identity_rows": 0,
                "identity_rows_without_people": 0,
            },
            "drift_counts": {
                "raw_rows_without_person": (
                    int(biometric_snapshot["raw_orphan_count"])
                    if biometric_snapshot["connection_ok"]
                    else None
                ),
                "vector_rows_without_person": (
                    int(biometric_snapshot["vector_orphan_count"])
                    if biometric_snapshot["connection_ok"]
                    else None
                ),
                "people_without_identity_rows": None,
                "identity_rows_without_people": None,
            },
        }
        if biometric_snapshot["connection_ok"] and biometric_snapshot["raw_orphan_sample"]:
            self._append_issue(
                issues,
                code="raw_rows_without_person",
                severity="warning",
                database_role="biometric_db",
                message=(
                    f"{self.raw_table} contains rows without a matching {self.person_table}; sample random_id values: "
                    + ", ".join(biometric_snapshot["raw_orphan_sample"])
                ),
                table=self.raw_table,
                row_count=int(biometric_snapshot["raw_orphan_count"]),
                sample_random_ids=list(biometric_snapshot["raw_orphan_sample"]),
            )
        if biometric_snapshot["connection_ok"] and biometric_snapshot["vector_orphan_sample"]:
            self._append_issue(
                issues,
                code="vector_rows_without_person",
                severity="warning",
                database_role="biometric_db",
                message=(
                    f"{self.vector_table} contains rows without a matching {self.person_table}; sample random_id values: "
                    + ", ".join(biometric_snapshot["vector_orphan_sample"])
                ),
                table=self.vector_table,
                row_count=int(biometric_snapshot["vector_orphan_count"]),
                sample_random_ids=list(biometric_snapshot["vector_orphan_sample"]),
            )

        people_count = biometric_snapshot["row_counts"]["people"]
        identity_count = identity_snapshot["row_counts"]["identity"]
        if people_count is None or identity_count is None:
            return reconciliation

        try:
            drift = self._inspect_identity_reconciliation_drift(
                people_count=int(people_count),
                identity_count=int(identity_count),
                sample_limit=RECONCILIATION_SAMPLE_LIMIT,
            )
        except Exception as exc:
            self._append_issue(
                issues,
                code="reconciliation_validation_failed",
                severity="error",
                database_role="identity_db" if self.dual_database_enabled else "biometric_db",
                message=(
                    "Bounded reconciliation validation failed while comparing "
                    f"{self.person_table} to {self.identity_table}: {exc}"
                ),
            )
            return reconciliation

        sample_counts = dict(reconciliation["sample_counts"])
        drift_counts = dict(reconciliation["drift_counts"])
        sample_counts["people_without_identity_rows"] = len(drift.people_without_identity_sample)
        sample_counts["identity_rows_without_people"] = len(drift.identity_without_people_sample)
        drift_counts["people_without_identity_rows"] = int(drift.people_without_identity_count)
        drift_counts["identity_rows_without_people"] = int(drift.identity_without_people_count)
        reconciliation["validation_mode"] = drift.validation_mode
        reconciliation["sample_counts"] = sample_counts
        reconciliation["drift_counts"] = drift_counts

        if drift.people_without_identity_sample:
            self._append_issue(
                issues,
                code="people_without_identity_rows",
                severity="warning",
                database_role="identity_db" if self.dual_database_enabled else "biometric_db",
                message=(
                    f"{self.person_table} contains rows without matching {self.identity_table} rows; sample random_id values: "
                    + ", ".join(drift.people_without_identity_sample)
                ),
                row_count=int(drift.people_without_identity_count),
                sample_random_ids=list(drift.people_without_identity_sample),
            )
        if drift.identity_without_people_sample:
            self._append_issue(
                issues,
                code="identity_rows_without_people",
                severity="warning",
                database_role="identity_db" if self.dual_database_enabled else "biometric_db",
                message=(
                    f"{self.identity_table} contains orphan rows without matching {self.person_table} rows; sample random_id values: "
                    + ", ".join(drift.identity_without_people_sample)
                ),
                row_count=int(drift.identity_without_people_count),
                sample_random_ids=list(drift.identity_without_people_sample),
            )

        if (
            drift.people_without_identity_count == 0
            and drift.identity_without_people_count == 0
            and int(people_count) != int(identity_count)
        ):
            self._append_issue(
                issues,
                code="people_identity_row_count_mismatch",
                severity="warning",
                database_role="identity_db" if self.dual_database_enabled else "biometric_db",
                message=(
                    f"{self.person_table} count ({people_count}) does not match {self.identity_table} count ({identity_count})."
                ),
                people_count=int(people_count),
                identity_count=int(identity_count),
            )
        return reconciliation

    def _build_schema_hardening_state(
        self,
        *,
        biometric_snapshot: Dict[str, Any],
        identity_snapshot: Dict[str, Any],
    ) -> Dict[str, object]:
        index_specs = self._schema_index_contract()
        constraint_specs = self._identity_constraint_contract()

        indexes: Dict[str, Dict[str, object]] = {}
        missing_indexes: List[str] = []
        for spec in index_specs:
            snapshot = biometric_snapshot if spec["database_role"] == "biometric_db" else identity_snapshot
            table_present = bool(snapshot["table_presence"].get(spec["table_key"]))
            present = bool(snapshot["index_presence"].get(spec["key"])) if table_present else False
            indexes[spec["key"]] = {
                "database_role": self._inspection_contract_database_role(str(spec["database_role"])),
                "table": self._logical_table_name(str(spec["table_key"])),
                "index_name": spec["name"],
                "present": present,
            }
            if table_present and not present:
                missing_indexes.append(spec["key"])

        constraints: Dict[str, Dict[str, object]] = {}
        missing_constraints: List[str] = []
        identity_table_present = bool(identity_snapshot["table_presence"].get("identity"))
        identity_storage_role = "identity_db" if self.dual_database_enabled else "biometric_db"
        for spec in constraint_specs:
            present = bool(identity_snapshot["identity_constraint_presence"].get(spec["key"])) if identity_table_present else False
            item: Dict[str, object] = {
                "database_role": identity_storage_role,
                "table": "identity_map",
                "present": present,
            }
            if spec["kind"] == "column_not_null":
                item["column"] = spec["column"]
            else:
                item["constraint_name"] = spec["name"]
            constraints[spec["key"]] = item
            if identity_table_present and not present:
                missing_constraints.append(spec["key"])

        required_columns = [
            spec["key"] for spec in constraint_specs if spec["kind"] == "column_not_null"
        ]
        named_checks = [
            spec["key"] for spec in constraint_specs if spec["kind"] == "check_constraint"
        ]
        legacy_schema_elements: List[str] = []
        if biometric_snapshot["legacy_person_columns"]["national_id"]:
            legacy_schema_elements.append(f"{self.person_table}.national_id")
        if biometric_snapshot["legacy_person_columns"]["full_name"]:
            legacy_schema_elements.append(f"{self.person_table}.full_name")
        if biometric_snapshot["legacy_person_columns"]["name_norm"]:
            legacy_schema_elements.append(f"{self.person_table}.name_norm")
        if self.dual_database_enabled and biometric_snapshot["table_presence"]["identity"]:
            legacy_schema_elements.append(f"biometric_db.{self.identity_table}")

        identity_rows_missing_profile_sample = list(identity_snapshot["identity_profile_missing_sample"])
        required_columns_enforced = all(bool(constraints[key]["present"]) for key in required_columns)
        checks_enforced = all(bool(constraints[key]["present"]) for key in named_checks)
        contract_enforced = identity_table_present and required_columns_enforced and checks_enforced
        profiles_complete = identity_table_present and not identity_rows_missing_profile_sample

        return {
            "indexes": indexes,
            "constraints": constraints,
            "identity_map_guarantees": {
                "database_role": identity_storage_role,
                "required_columns_enforced": required_columns_enforced,
                "checks_enforced": checks_enforced,
                "profiles_complete": profiles_complete,
                "contract_enforced": contract_enforced,
                "completeness_guaranteed": contract_enforced and profiles_complete,
            },
            "drift": {
                "schema_drift_detected": bool(
                    missing_indexes or missing_constraints or legacy_schema_elements or identity_rows_missing_profile_sample
                ),
                "missing_indexes": missing_indexes,
                "missing_constraints": missing_constraints,
                "legacy_schema_elements": legacy_schema_elements,
                "identity_rows_missing_profile_sample": identity_rows_missing_profile_sample,
            },
            "contract_descriptions": {
                "indexes": {
                    spec["key"]: {
                        "database_role": self._inspection_contract_database_role(str(spec["database_role"])),
                        "table": self._logical_table_name(str(spec["table_key"])),
                        "index_name": spec["name"],
                    }
                    for spec in index_specs
                },
                "constraints": {
                    key: {
                        "database_role": identity_storage_role,
                        "table": "identity_map",
                        "kind": str(spec["kind"]),
                        "column": spec.get("column"),
                        "constraint_name": spec.get("name"),
                    }
                    for key, spec in {
                        spec["key"]: spec for spec in constraint_specs
                    }.items()
                },
            },
        }

    def _append_schema_hardening_issues(
        self,
        issues: List[Dict[str, object]],
        *,
        biometric_snapshot: Dict[str, Any],
        identity_snapshot: Dict[str, Any],
        schema_hardening: Dict[str, object],
    ) -> None:
        drift = dict(schema_hardening.get("drift", {}))
        missing_constraints = list(drift.get("missing_constraints", []))
        if missing_constraints and identity_snapshot["connection_ok"] and identity_snapshot["table_presence"]["identity"]:
            self._append_issue(
                issues,
                code="identity_schema_contract_missing",
                severity="error",
                database_role="identity_db" if self.dual_database_enabled else "biometric_db",
                message=(
                    f"{self.identity_table} is missing hardened SQL contract items: "
                    + ", ".join(missing_constraints)
                ),
                table=self.identity_table,
                missing_items=missing_constraints,
            )

        missing_indexes = list(drift.get("missing_indexes", []))
        if not missing_indexes:
            return

        index_specs_by_key = {spec["key"]: spec for spec in self._schema_index_contract()}
        for database_role in ("biometric_db", "identity_db"):
            role_missing = [
                key
                for key in missing_indexes
                if self._inspection_contract_database_role(str(index_specs_by_key[key]["database_role"])) == database_role
            ]
            if not role_missing:
                continue
            snapshot = biometric_snapshot if database_role == "biometric_db" else identity_snapshot
            if not snapshot["connection_ok"]:
                continue
            self._append_issue(
                issues,
                code="supporting_indexes_missing",
                severity="warning",
                database_role=database_role,
                message=(
                    f"{database_role} is missing supporting indexes for the hardened schema contract: "
                    + ", ".join(role_missing)
                ),
                missing_items=role_missing,
            )

    def _reconciliation_script_command(self, *flags: str) -> str:
        parts = ["python", "scripts/diagnostics/reconcile_identification_runtime_db.py"]
        table_prefix = str(getattr(self, "table_prefix", "") or "")
        if table_prefix:
            parts.extend(["--table-prefix", table_prefix])
        parts.extend(str(flag) for flag in flags if str(flag).strip())
        return " ".join(parts)

    def _reconciliation_command_examples(self) -> Dict[str, str]:
        return {
            "report_only": self._reconciliation_script_command(),
            "repair_raw_orphans": self._reconciliation_script_command("--repair-raw-orphans"),
            "repair_vector_orphans": self._reconciliation_script_command("--repair-vector-orphans"),
            "repair_identity_orphans": self._reconciliation_script_command("--repair-identity-orphans"),
            "redact_legacy_raw_image_bytes": self._reconciliation_script_command(
                "--redact-legacy-raw-image-bytes"
            ),
            "backfill_generic_retrieval_vectors": self._reconciliation_script_command(
                "--backfill-generic-retrieval-vectors"
            ),
        }

    def _issue_reconciliation_guidance(self, code: str) -> Dict[str, object]:
        report_cmd = self._reconciliation_script_command()
        guidance: Dict[str, Dict[str, object]] = {
            "raw_rows_without_person": {
                "repairability": "safely_repairable",
                "manual_reconciliation_required": False,
                "repair_actions": ["repair_raw_orphans"],
                "repair_commands": [self._reconciliation_script_command("--repair-raw-orphans")],
                "remediation": (
                    "Safe repair is available because the orphan raw_fingerprints rows can be proven from current "
                    "biometric_db state. Run "
                    + self._reconciliation_script_command("--repair-raw-orphans")
                    + "."
                ),
            },
            "vector_rows_without_person": {
                "repairability": "safely_repairable",
                "manual_reconciliation_required": False,
                "repair_actions": ["repair_vector_orphans"],
                "repair_commands": [self._reconciliation_script_command("--repair-vector-orphans")],
                "remediation": (
                    "Safe repair is available because the orphan feature_vectors rows can be proven from current "
                    "biometric_db state. Run "
                    + self._reconciliation_script_command("--repair-vector-orphans")
                    + "."
                ),
            },
            "identity_rows_without_people": {
                "repairability": "safely_repairable",
                "manual_reconciliation_required": False,
                "repair_actions": ["repair_identity_orphans"],
                "repair_commands": [self._reconciliation_script_command("--repair-identity-orphans")],
                "remediation": (
                    "Safe repair is available only behind an explicit flag because the orphan identity_map rows "
                    "cannot be matched to person_directory rows in the current runtime state. Run "
                    + self._reconciliation_script_command("--repair-identity-orphans")
                    + " only if deleting those orphan identity rows is acceptable."
                ),
            },
            "legacy_raw_image_bytes_present": {
                "repairability": "safely_repairable",
                "manual_reconciliation_required": False,
                "repair_actions": ["redact_legacy_raw_image_bytes"],
                "repair_commands": [self._reconciliation_script_command("--redact-legacy-raw-image-bytes")],
                "remediation": (
                    "Safe redaction is available because only legacy raw biometric image payloads are cleared; "
                    "metadata, hashes, vectors, and identity rows are preserved. Run "
                    + self._reconciliation_script_command("--redact-legacy-raw-image-bytes")
                    + "."
                ),
            },
            "generic_vector_backfill_recommended": {
                "repairability": "safely_repairable",
                "manual_reconciliation_required": False,
                "repair_actions": ["backfill_generic_retrieval_vectors"],
                "repair_commands": [self._reconciliation_script_command("--backfill-generic-retrieval-vectors")],
                "remediation": (
                    "Safe backfill is available because existing feature_vectors rows can be copied into "
                    "method_retrieval_vectors without reading raw images or re-embedding. Run "
                    + self._reconciliation_script_command("--backfill-generic-retrieval-vectors")
                    + "."
                ),
            },
            "people_without_identity_rows": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Manual reconciliation is required because full_name, name_norm, and national_id cannot be "
                    "reconstructed safely from biometric_db alone. Restore the missing identity_map rows from the "
                    "authoritative source or re-enroll/purge the affected person_directory rows after review. "
                    f"Start with {report_cmd}."
                ),
            },
            "identity_rows_missing_profile": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Manual reconciliation is required because missing full_name/name_norm values cannot be guessed "
                    "safely. Restore the profile fields from the authoritative identity source, then rerun "
                    f"{report_cmd}."
                ),
            },
            "database_connection_failed": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Fix connectivity or credentials for the affected database, then rerun "
                    f"{report_cmd}."
                ),
            },
            "reconciliation_validation_failed": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "The bounded reconciliation comparison did not complete cleanly. Resolve the underlying error "
                    f"and rerun {report_cmd}."
                ),
            },
            "missing_table": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Restore or bootstrap the missing table on the correct runtime database, then rerun "
                    f"{report_cmd}."
                ),
            },
            "legacy_person_national_id_column_present": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Re-run the schema migration/hardening flow for this table prefix and verify that the legacy "
                    f"column is removed. Then rerun {report_cmd}."
                ),
            },
            "legacy_person_profile_columns_present": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Re-run the schema migration/hardening flow for this table prefix and verify that the legacy "
                    f"profile columns are removed. Then rerun {report_cmd}."
                ),
            },
            "legacy_identity_table_present_in_biometric_db": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Review whether the legacy biometric_db identity_map table still contains source-of-truth data. "
                    f"Only remove it during a planned migration window, then rerun {report_cmd}."
                ),
            },
            "missing_vector_extension": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Install pgvector in biometric_db and rerun store bootstrap/validation, then rerun "
                    f"{report_cmd}."
                ),
            },
            "unexpected_vector_methods_present": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Review and clean the unexpected feature_vectors methods manually if they are not part of the "
                    f"supported retrieval contract, then rerun {report_cmd}."
                ),
            },
            "people_identity_row_count_mismatch": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "The person_directory and identity_map counts diverged, but an exact bounded mismatch sample was "
                    f"not available. Investigate with {report_cmd} before applying any manual cleanup."
                ),
            },
            "identity_schema_contract_missing": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Restore the hardened identity_map SQL contract before serving traffic, then rerun "
                    f"{report_cmd}."
                ),
            },
            "supporting_indexes_missing": {
                "repairability": "not_safely_repairable",
                "manual_reconciliation_required": True,
                "repair_actions": [],
                "repair_commands": [],
                "remediation": (
                    "Recreate the missing supporting indexes during a maintenance window, then rerun "
                    f"{report_cmd}."
                ),
            },
        }
        return dict(guidance.get(code, {
            "repairability": "not_safely_repairable",
            "manual_reconciliation_required": True,
            "repair_actions": [],
            "repair_commands": [],
            "remediation": f"Review the issue manually and rerun {report_cmd}.",
        }))

    def _summarize_issue_repairability(self, issues: Sequence[Dict[str, object]]) -> Dict[str, object]:
        safe_issue_codes = sorted(
            {
                str(issue["code"])
                for issue in issues
                if str(issue.get("repairability")) == "safely_repairable"
            }
        )
        manual_issue_codes = sorted(
            {
                str(issue["code"])
                for issue in issues
                if str(issue.get("repairability")) != "safely_repairable"
            }
        )
        return {
            "safely_repairable_issue_count": len(safe_issue_codes),
            "not_safely_repairable_issue_count": len(manual_issue_codes),
            "safely_repairable_issue_codes": safe_issue_codes,
            "not_safely_repairable_issue_codes": manual_issue_codes,
        }

    def _build_reconciliation_summary(self, issues: Sequence[Dict[str, object]]) -> Dict[str, object]:
        severity_counts = {
            "informational": 0,
            "warning": 0,
            "error": 0,
        }
        repairability_counts = {
            "safely_repairable": 0,
            "not_safely_repairable": 0,
        }
        manual_reconciliation_required = False
        for issue in issues:
            severity = str(issue.get("severity", "")).lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
            else:
                severity_counts["informational"] += 1

            repairability = str(issue.get("repairability", "not_safely_repairable"))
            if repairability not in repairability_counts:
                repairability = "not_safely_repairable"
            repairability_counts[repairability] += 1
            manual_reconciliation_required = manual_reconciliation_required or bool(
                issue.get("manual_reconciliation_required")
            )

        return {
            "severity": severity_counts,
            "repairability": repairability_counts,
            "manual_reconciliation_required": manual_reconciliation_required,
        }

    def _append_issue(
        self,
        issues: List[Dict[str, object]],
        *,
        code: str,
        severity: str,
        database_role: str,
        message: str,
        **details: object,
    ) -> None:
        issue: Dict[str, object] = {
            "code": code,
            "severity": severity,
            "database_role": database_role,
            "message": message,
        }
        for key, value in self._issue_reconciliation_guidance(code).items():
            if value is not None:
                issue[key] = value
        for key, value in details.items():
            if value is not None:
                issue[key] = value
        issues.append(issue)

    def _connect_postgres_inspection(self, database_url: str, *, database_role: str):
        psycopg, dict_row = _load_postgres_base_deps()
        try:
            return psycopg.connect(
                database_url,
                row_factory=dict_row,
                options="-c default_transaction_read_only=on",
            )
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                f"Failed to connect to {database_role} database in read-only inspection mode "
                f"({self._redacted_database_url(database_url)}): {exc}"
            ) from exc

    def _inspection_contract_database_role(self, database_role: str) -> str:
        if not self.dual_database_enabled and database_role == "identity_db":
            return "biometric_db"
        return database_role

    @staticmethod
    def _logical_table_name(table_key: str) -> str:
        mapping = {
            "person": "person_directory",
            "identity": "identity_map",
            "raw": "raw_fingerprints",
            "vectors": "feature_vectors",
            "generic_vectors": "method_retrieval_vectors",
        }
        return mapping.get(table_key, table_key)

    @staticmethod
    def _count_rows(cur, table_name: str) -> int:
        cur.execute(f"SELECT COUNT(*) AS n FROM {table_name}")
        row = cur.fetchone()
        return 0 if row is None else int(row["n"])

    def _count_vector_rows_by_method(self, cur) -> Dict[str, int]:
        cur.execute(
            f"""
            SELECT method, COUNT(*) AS n
            FROM {self.vector_table}
            GROUP BY method
            ORDER BY method
            """
        )
        return {str(row["method"]): int(row["n"]) for row in cur.fetchall()}

    @staticmethod
    def _generic_vector_count_key(*, method: str, vector_kind: str) -> str:
        return f"{method}/{vector_kind}"

    def _count_generic_vector_rows_by_method_kind(self, cur) -> Dict[str, int]:
        cur.execute(
            f"""
            SELECT method, vector_kind, COUNT(*) AS n
            FROM {self.generic_vector_table}
            GROUP BY method, vector_kind
            ORDER BY method, vector_kind
            """
        )
        return {
            self._generic_vector_count_key(
                method=str(row["method"]),
                vector_kind=str(row["vector_kind"]),
            ): int(row["n"])
            for row in cur.fetchall()
        }

    def _generic_vector_schema_compatible(self, cur) -> bool:
        required_columns = {
            "random_id",
            "method",
            "vector_kind",
            "dim",
            "distance_metric",
            "created_at",
            "metadata_json",
            "vector_512",
            "vector_768",
        }
        columns_present = all(
            self._column_exists(cur, self.generic_vector_table, column)
            for column in required_columns
        )
        return columns_present and self._generic_vector_constraints_compatible(cur)

    def _generic_vector_constraints_compatible(self, cur) -> bool:
        cur.execute(
            """
            SELECT c.contype, pg_get_constraintdef(c.oid) AS definition
            FROM pg_catalog.pg_constraint AS c
            JOIN pg_catalog.pg_class AS t ON t.oid = c.conrelid
            JOIN pg_catalog.pg_namespace AS n ON n.oid = t.relnamespace
            WHERE n.nspname = current_schema()
              AND t.relname = %s
            """,
            (self.generic_vector_table,),
        )
        definitions = [
            (str(row["contype"]), str(row["definition"]).lower())
            for row in cur.fetchall()
        ]
        has_primary_key = any(
            contype == "p"
            and "random_id" in definition
            and "method" in definition
            and "vector_kind" in definition
            for contype, definition in definitions
        )
        has_method_generic_dim_check = any(
            contype == "c"
            and "dim = 512" in definition
            and "dim = 768" in definition
            and "method =" not in definition
            and "method=" not in definition
            for contype, definition in definitions
        )
        return has_primary_key and has_method_generic_dim_check

    @staticmethod
    def _count_orphan_rows(cur, *, child_table: str, parent_table: str) -> int:
        cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {child_table} AS c
            WHERE NOT EXISTS (
                SELECT 1
                FROM {parent_table} AS p
                WHERE p.random_id = c.random_id
            )
            """
        )
        row = cur.fetchone()
        return 0 if row is None else int(row["n"])

    def _count_identity_rows_missing_profile(self, cur) -> int:
        cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {self.identity_table}
            WHERE full_name IS NULL OR BTRIM(full_name) = ''
               OR name_norm IS NULL OR BTRIM(name_norm) = ''
            """
        )
        row = cur.fetchone()
        return 0 if row is None else int(row["n"])

    def _count_legacy_raw_image_bytes_rows(self, cur) -> int:
        cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {self.raw_table}
            WHERE image_bytes IS NOT NULL
            """
        )
        row = cur.fetchone()
        return 0 if row is None else int(row["n"])

    def _sample_legacy_raw_image_bytes_random_ids(
        self,
        cur,
        *,
        limit: int = RECONCILIATION_SAMPLE_LIMIT,
    ) -> List[str]:
        cur.execute(
            f"""
            SELECT random_id
            FROM {self.raw_table}
            WHERE image_bytes IS NOT NULL
            ORDER BY random_id
            LIMIT %s
            """,
            (int(limit),),
        )
        return [str(row["random_id"]) for row in cur.fetchall()]

    @staticmethod
    def _sample_missing_random_ids(
        cur,
        *,
        source_table: str,
        reference_table: str,
        limit: int = RECONCILIATION_SAMPLE_LIMIT,
    ) -> List[str]:
        cur.execute(
            f"""
            SELECT s.random_id
            FROM {source_table} AS s
            WHERE NOT EXISTS (
                SELECT 1
                FROM {reference_table} AS r
                WHERE r.random_id = s.random_id
            )
            ORDER BY s.random_id
            LIMIT %s
            """,
            (int(limit),),
        )
        return [str(row["random_id"]) for row in cur.fetchall()]

    @staticmethod
    def _count_missing_random_ids(cur, *, source_table: str, reference_table: str) -> int:
        cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {source_table} AS s
            WHERE NOT EXISTS (
                SELECT 1
                FROM {reference_table} AS r
                WHERE r.random_id = s.random_id
            )
            """
        )
        row = cur.fetchone()
        return 0 if row is None else int(row["n"])

    @staticmethod
    def _sample_orphan_ids(cur, *, child_table: str, parent_table: str, limit: int = 10) -> List[str]:
        cur.execute(
            f"""
            SELECT c.random_id
            FROM {child_table} AS c
            LEFT JOIN {parent_table} AS p ON p.random_id = c.random_id
            WHERE p.random_id IS NULL
            ORDER BY c.random_id
            LIMIT %s
            """,
            (int(limit),),
        )
        return [str(row["random_id"]) for row in cur.fetchall()]

    def _inspect_identity_reconciliation_drift(
        self,
        *,
        people_count: int,
        identity_count: int,
        sample_limit: int = RECONCILIATION_SAMPLE_LIMIT,
    ) -> _IdentityReconciliationDrift:
        if self.dual_database_enabled:
            return self._inspect_dual_database_identity_drift(
                people_count=people_count,
                identity_count=identity_count,
                sample_limit=sample_limit,
            )
        return self._inspect_single_database_identity_drift(
            people_count=people_count,
            identity_count=identity_count,
            sample_limit=sample_limit,
        )

    def _inspect_single_database_identity_drift(
        self,
        *,
        people_count: int,
        identity_count: int,
        sample_limit: int,
    ) -> _IdentityReconciliationDrift:
        with self._connect_postgres_inspection(
            self.biometric_database_url,
            database_role="biometric_db",
        ) as conn:
            with conn.cursor() as cur:
                people_without_identity_sample = self._sample_missing_random_ids(
                    cur,
                    source_table=self.person_table,
                    reference_table=self.identity_table,
                    limit=sample_limit,
                )
                identity_without_people_sample = self._sample_missing_random_ids(
                    cur,
                    source_table=self.identity_table,
                    reference_table=self.person_table,
                    limit=sample_limit,
                )
                people_without_identity_count = self._count_missing_random_ids(
                    cur,
                    source_table=self.person_table,
                    reference_table=self.identity_table,
                )
                identity_without_people_count = self._count_missing_random_ids(
                    cur,
                    source_table=self.identity_table,
                    reference_table=self.person_table,
                )

        return _IdentityReconciliationDrift(
            validation_mode="bounded_sql_antijoin",
            sample_limit=int(sample_limit),
            people_count=int(people_count),
            identity_count=int(identity_count),
            people_without_identity_count=int(people_without_identity_count),
            people_without_identity_sample=people_without_identity_sample,
            identity_without_people_count=int(identity_without_people_count),
            identity_without_people_sample=identity_without_people_sample,
        )

    def _inspect_dual_database_identity_drift(
        self,
        *,
        people_count: int,
        identity_count: int,
        sample_limit: int,
    ) -> _IdentityReconciliationDrift:
        people_without_identity_count = 0
        identity_without_people_count = 0
        people_without_identity_sample: List[str] = []
        identity_without_people_sample: List[str] = []

        with self._connect_postgres_inspection(
            self.biometric_database_url,
            database_role="biometric_db",
        ) as bio_conn:
            with self._connect_postgres_inspection(
                self.identity_database_url,
                database_role="identity_db",
            ) as ident_conn:
                people_ids = self._iter_random_ids_streaming(
                    bio_conn,
                    table_name=self.person_table,
                    cursor_tag="people_scan",
                )
                identity_ids = self._iter_random_ids_streaming(
                    ident_conn,
                    table_name=self.identity_table,
                    cursor_tag="identity_scan",
                )
                for mismatch_kind, random_id in self._stream_sorted_random_id_differences(
                    people_ids,
                    identity_ids,
                ):
                    if mismatch_kind == "people_without_identity":
                        people_without_identity_count += 1
                        if len(people_without_identity_sample) < int(sample_limit):
                            people_without_identity_sample.append(random_id)
                        continue
                    identity_without_people_count += 1
                    if len(identity_without_people_sample) < int(sample_limit):
                        identity_without_people_sample.append(random_id)

        return _IdentityReconciliationDrift(
            validation_mode="bounded_exact_streaming",
            sample_limit=int(sample_limit),
            people_count=int(people_count),
            identity_count=int(identity_count),
            people_without_identity_count=int(people_without_identity_count),
            people_without_identity_sample=people_without_identity_sample,
            identity_without_people_count=int(identity_without_people_count),
            identity_without_people_sample=identity_without_people_sample,
        )

    def _iter_random_ids_streaming(
        self,
        conn,
        *,
        table_name: str,
        cursor_tag: str,
        batch_size: int = RECONCILIATION_STREAM_BATCH_SIZE,
    ) -> Iterator[str]:
        cursor_name = self._stream_cursor_name(cursor_tag)
        try:
            cur = conn.cursor(name=cursor_name)
        except TypeError:
            cur = conn.cursor()

        with cur:
            cur.execute(f"SELECT random_id FROM {table_name} ORDER BY random_id")
            while True:
                if hasattr(cur, "fetchmany"):
                    rows = cur.fetchmany(int(batch_size))
                else:  # pragma: no cover - compatibility fallback
                    rows = cur.fetchall()
                if not rows:
                    break
                for row in rows:
                    yield str(row["random_id"])
                if not hasattr(cur, "fetchmany"):  # pragma: no cover - compatibility fallback
                    break

    def _stream_cursor_name(self, cursor_tag: str) -> str:
        prefix = (self.table_prefix or "identify")[:16]
        tag = re.sub(r"[^a-z0-9_]+", "_", str(cursor_tag).lower())[:20]
        return f"{prefix}_{tag}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _stream_sorted_random_id_differences(
        people_ids: Iterator[str],
        identity_ids: Iterator[str],
    ) -> Iterator[tuple[str, str]]:
        current_people = SecureSplitFingerprintStore._next_stream_random_id(people_ids)
        current_identity = SecureSplitFingerprintStore._next_stream_random_id(identity_ids)

        while current_people is not None or current_identity is not None:
            if current_identity is None or (
                current_people is not None and current_people < current_identity
            ):
                yield ("people_without_identity", current_people)
                current_people = SecureSplitFingerprintStore._next_stream_random_id(people_ids)
                continue
            if current_people is None or current_identity < current_people:
                yield ("identity_without_people", current_identity)
                current_identity = SecureSplitFingerprintStore._next_stream_random_id(identity_ids)
                continue

            current_people = SecureSplitFingerprintStore._next_stream_random_id(people_ids)
            current_identity = SecureSplitFingerprintStore._next_stream_random_id(identity_ids)

    @staticmethod
    def _next_stream_random_id(random_ids: Iterator[str]) -> Optional[str]:
        try:
            return str(next(random_ids))
        except StopIteration:
            return None

    @staticmethod
    def _has_vector_extension(cur) -> bool:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_extension
                WHERE extname = 'vector'
            ) AS has_vector
            """
        )
        row = cur.fetchone()
        return bool(row and row["has_vector"])

    def _normalize_repair_actions(self, repair_actions: Sequence[str] | None) -> List[str]:
        allowed = {
            "repair_raw_orphans",
            "repair_vector_orphans",
            "repair_identity_orphans",
            "redact_legacy_raw_image_bytes",
            "backfill_generic_retrieval_vectors",
        }
        normalized: List[str] = []
        for action in repair_actions or ():
            action_norm = str(action or "").strip().lower()
            if not action_norm:
                continue
            if action_norm not in allowed:
                raise ValueError(
                    "Unsupported reconciliation repair action: "
                    f"{action_norm}. Supported actions: {', '.join(sorted(allowed))}"
                )
            if action_norm not in normalized:
                normalized.append(action_norm)
        return normalized

    def _available_repair_actions(self, issues: Sequence[Dict[str, object]]) -> List[str]:
        available: List[str] = []
        for issue in issues:
            for action in issue.get("repair_actions", []):
                action_norm = str(action)
                if action_norm and action_norm not in available:
                    available.append(action_norm)
        return available

    def _apply_requested_repairs(self, repair_actions: Sequence[str]) -> List[Dict[str, object]]:
        repair_map = {
            "repair_raw_orphans": self._repair_raw_orphans,
            "repair_vector_orphans": self._repair_vector_orphans,
            "repair_identity_orphans": self._repair_identity_orphans,
            "redact_legacy_raw_image_bytes": self._redact_legacy_raw_image_bytes,
            "backfill_generic_retrieval_vectors": self._backfill_generic_retrieval_vectors,
        }
        results: List[Dict[str, object]] = []
        for action in repair_actions:
            result = dict(repair_map[str(action)]())
            result["requested_action"] = str(action)
            results.append(result)
        return results

    def _repair_raw_orphans(self) -> Dict[str, object]:
        return self._repair_same_database_orphans(
            child_table=self.raw_table,
            parent_table=self.person_table,
            database_role="biometric_db",
            action="repair_raw_orphans",
            command_flag="--repair-raw-orphans",
        )

    def _repair_vector_orphans(self) -> Dict[str, object]:
        return self._repair_same_database_orphans(
            child_table=self.vector_table,
            parent_table=self.person_table,
            database_role="biometric_db",
            action="repair_vector_orphans",
            command_flag="--repair-vector-orphans",
        )

    def _repair_identity_orphans(self) -> Dict[str, object]:
        if not self.dual_database_enabled:
            return self._repair_same_database_orphans(
                child_table=self.identity_table,
                parent_table=self.person_table,
                database_role="biometric_db",
                action="repair_identity_orphans",
                command_flag="--repair-identity-orphans",
            )
        return self._repair_dual_database_identity_orphans()

    def _redact_legacy_raw_image_bytes(self) -> Dict[str, object]:
        conn = self._connect_biometric()
        table_present = False
        column_present = False
        candidate_row_count = 0
        redacted_count = 0

        try:
            with conn.cursor() as cur:
                table_present = self._table_exists(cur, self.raw_table)
                if table_present:
                    column_present = self._column_exists(cur, self.raw_table, "image_bytes")
                if table_present and column_present:
                    candidate_row_count = self._count_legacy_raw_image_bytes_rows(cur)
                    cur.execute(
                        f"""
                        UPDATE {self.raw_table}
                        SET image_bytes = NULL
                        WHERE image_bytes IS NOT NULL
                        """
                    )
                    redacted_count = int(cur.rowcount or 0)
            conn.commit()
        except Exception:
            self._safe_rollback(conn)
            raise
        finally:
            conn.close()

        noop_reason = None
        status = "applied" if redacted_count else "noop"
        if not table_present:
            noop_reason = "raw_table_missing"
        elif not column_present:
            noop_reason = "image_bytes_column_missing"
        elif candidate_row_count == 0:
            noop_reason = "no_legacy_raw_image_bytes"

        result = {
            "action": "redact_legacy_raw_image_bytes",
            "database_role": "biometric_db",
            "status": status,
            "repair_command": self._reconciliation_script_command("--redact-legacy-raw-image-bytes"),
            "candidate_row_count": int(candidate_row_count),
            "redacted_count": int(redacted_count),
            "table": self.raw_table,
            "column": "image_bytes",
            "sensitive_backup_created": False,
        }
        if noop_reason is not None:
            result["noop_reason"] = noop_reason
        return result

    def _backfill_generic_retrieval_vectors(self) -> Dict[str, object]:
        conn = self._connect_biometric()
        candidate_row_count = 0
        copied_count = 0
        skipped_count = 0
        method_results: Dict[str, Dict[str, object]] = {}
        errors: List[Dict[str, str]] = []
        noop_reason = None

        try:
            with conn.cursor() as cur:
                self._require_vector_extension(cur)
                person_table_present = self._table_exists(cur, self.person_table)
                legacy_table_present = self._table_exists(cur, self.vector_table)
                if not person_table_present:
                    noop_reason = "person_table_missing"
                elif not legacy_table_present:
                    noop_reason = "legacy_feature_vectors_table_missing"

                if noop_reason is None:
                    self._init_generic_vector_schema(cur)
                    for method in sorted(_legacy_storage_methods()):
                        spec = VECTOR_SPECS[method]
                        cur.execute("SAVEPOINT backfill_generic_retrieval_vectors_method")
                        try:
                            legacy_count = self._count_legacy_vector_rows_for_spec(cur, spec=spec)
                            existing_before = self._count_generic_vector_rows_for_spec(cur, spec=spec)
                            metadata = json.dumps(
                                {
                                    "source": "legacy_feature_vectors",
                                    "backfilled": True,
                                }
                            )
                            cur.execute(
                                f"""
                                INSERT INTO {self.generic_vector_table}
                                    (
                                        random_id,
                                        method,
                                        vector_kind,
                                        dim,
                                        distance_metric,
                                        created_at,
                                        metadata_json,
                                        vector_512,
                                        vector_768,
                                        vector_blob
                                    )
                                SELECT
                                    random_id,
                                    method,
                                    %s,
                                    dim,
                                    %s,
                                    created_at,
                                    %s::jsonb,
                                    vector_512,
                                    vector_768,
                                    NULL
                                FROM {self.vector_table}
                                WHERE method = %s
                                  AND dim = %s
                                ON CONFLICT (random_id, method, vector_kind) DO NOTHING
                                """,
                                (spec.vector_kind, spec.distance_metric, metadata, method, spec.dim),
                            )
                            copied = int(cur.rowcount or 0)
                            existing_after = self._count_generic_vector_rows_for_spec(cur, spec=spec)
                            skipped = max(0, int(legacy_count) - copied)
                            candidate_row_count += int(legacy_count)
                            copied_count += copied
                            skipped_count += skipped
                            method_results[method] = {
                                "legacy_row_count": int(legacy_count),
                                "generic_existing_before": int(existing_before),
                                "generic_existing_after": int(existing_after),
                                "copied": copied,
                                "skipped": skipped,
                                "vector_kind": spec.vector_kind,
                                "dim": spec.dim,
                                "distance_metric": spec.distance_metric,
                            }
                            cur.execute("RELEASE SAVEPOINT backfill_generic_retrieval_vectors_method")
                        except Exception as exc:
                            cur.execute("ROLLBACK TO SAVEPOINT backfill_generic_retrieval_vectors_method")
                            cur.execute("RELEASE SAVEPOINT backfill_generic_retrieval_vectors_method")
                            errors.append({"method": method, "error": str(exc)})
            conn.commit()
        except Exception:
            self._safe_rollback(conn)
            raise
        finally:
            conn.close()

        status = "applied" if copied_count else "noop"
        if errors:
            status = "partial" if copied_count else "failed"
        elif noop_reason is None and candidate_row_count == 0:
            noop_reason = "no_legacy_vectors"

        result: Dict[str, object] = {
            "action": "backfill_generic_retrieval_vectors",
            "database_role": "biometric_db",
            "status": status,
            "repair_command": self._reconciliation_script_command("--backfill-generic-retrieval-vectors"),
            "source_table": self.vector_table,
            "target_table": self.generic_vector_table,
            "candidate_row_count": int(candidate_row_count),
            "copied_count": int(copied_count),
            "skipped_count": int(skipped_count),
            "error_count": len(errors),
            "errors": errors,
            "methods": method_results,
        }
        if noop_reason is not None:
            result["noop_reason"] = noop_reason
        return result

    def _repair_same_database_orphans(
        self,
        *,
        child_table: str,
        parent_table: str,
        database_role: str,
        action: str,
        command_flag: str,
    ) -> Dict[str, object]:
        conn = self._connect_biometric()
        orphan_count = 0
        sample_random_ids: List[str] = []
        deleted_rows = 0

        try:
            with conn.cursor() as cur:
                orphan_count = self._count_orphan_rows(
                    cur,
                    child_table=child_table,
                    parent_table=parent_table,
                )
                sample_random_ids = self._sample_orphan_ids(
                    cur,
                    child_table=child_table,
                    parent_table=parent_table,
                    limit=RECONCILIATION_SAMPLE_LIMIT,
                )
                cur.execute(
                    f"""
                    DELETE FROM {child_table} AS c
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM {parent_table} AS p
                        WHERE p.random_id = c.random_id
                    )
                    """
                )
                deleted_rows = int(cur.rowcount or 0)
            conn.commit()
        except Exception:
            self._safe_rollback(conn)
            raise
        finally:
            conn.close()

        return {
            "action": action,
            "database_role": database_role,
            "status": "applied",
            "repair_command": self._reconciliation_script_command(command_flag),
            "candidate_row_count": int(orphan_count),
            "deleted_rows": int(deleted_rows),
            "sample_random_ids": sample_random_ids,
        }

    def _repair_dual_database_identity_orphans(self) -> Dict[str, object]:
        write_conn = self._connect_identity()
        candidate_row_count = 0
        candidate_sample_random_ids: List[str] = []
        deleted_sample_random_ids: List[str] = []
        deleted_rows = 0
        batch: List[str] = []

        try:
            for random_id in self._iter_dual_database_identity_orphan_ids():
                candidate_row_count += 1
                if len(candidate_sample_random_ids) < RECONCILIATION_SAMPLE_LIMIT:
                    candidate_sample_random_ids.append(random_id)
                batch.append(random_id)
                if len(batch) >= RECONCILIATION_STREAM_BATCH_SIZE:
                    deleted = self._delete_verified_identity_orphan_batch(write_conn, batch)
                    deleted_rows += len(deleted)
                    self._extend_unique_sample(deleted_sample_random_ids, deleted, RECONCILIATION_SAMPLE_LIMIT)
                    batch.clear()

            if batch:
                deleted = self._delete_verified_identity_orphan_batch(write_conn, batch)
                deleted_rows += len(deleted)
                self._extend_unique_sample(deleted_sample_random_ids, deleted, RECONCILIATION_SAMPLE_LIMIT)
                batch.clear()

            write_conn.commit()
        except Exception:
            self._safe_rollback(write_conn)
            raise
        finally:
            write_conn.close()

        return {
            "action": "repair_identity_orphans",
            "database_role": "identity_db",
            "status": "applied",
            "repair_command": self._reconciliation_script_command("--repair-identity-orphans"),
            "candidate_row_count": int(candidate_row_count),
            "deleted_rows": int(deleted_rows),
            "sample_random_ids": candidate_sample_random_ids,
            "deleted_sample_random_ids": deleted_sample_random_ids,
        }

    def _iter_dual_database_identity_orphan_ids(self) -> Iterator[str]:
        with self._connect_postgres_inspection(
            self.biometric_database_url,
            database_role="biometric_db",
        ) as bio_conn:
            with self._connect_postgres_inspection(
                self.identity_database_url,
                database_role="identity_db",
            ) as ident_conn:
                people_ids = self._iter_random_ids_streaming(
                    bio_conn,
                    table_name=self.person_table,
                    cursor_tag="people_orphan_repair",
                )
                identity_ids = self._iter_random_ids_streaming(
                    ident_conn,
                    table_name=self.identity_table,
                    cursor_tag="identity_orphan_repair",
                )
                for mismatch_kind, random_id in self._stream_sorted_random_id_differences(
                    people_ids,
                    identity_ids,
                ):
                    if mismatch_kind == "identity_without_people":
                        yield random_id

    def _delete_verified_identity_orphan_batch(self, write_conn, candidate_ids: Sequence[str]) -> List[str]:
        if not candidate_ids:
            return []
        existing_people = self._fetch_existing_person_ids(candidate_ids)
        delete_ids = [str(random_id) for random_id in candidate_ids if str(random_id) not in existing_people]
        if not delete_ids:
            return []

        with write_conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.identity_table} WHERE random_id = ANY(%s)",
                (list(delete_ids),),
            )
        return delete_ids

    def _fetch_existing_person_ids(self, random_ids: Sequence[str]) -> set[str]:
        if not random_ids:
            return set()
        with self._connect_postgres_inspection(
            self.biometric_database_url,
            database_role="biometric_db",
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT random_id
                    FROM {self.person_table}
                    WHERE random_id = ANY(%s)
                    ORDER BY random_id
                    """,
                    (list(random_ids),),
                )
                return {str(row["random_id"]) for row in cur.fetchall()}

    @staticmethod
    def _extend_unique_sample(target: List[str], values: Sequence[str], limit: int) -> None:
        for value in values:
            value_text = str(value)
            if value_text in target:
                continue
            target.append(value_text)
            if len(target) >= int(limit):
                break

    # ------------------------------------------------------------------
    # initialization + migration
    # ------------------------------------------------------------------
    def _init_all(self) -> None:
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                self._init_biometric_schema(cur)

        if self.dual_database_enabled:
            with self._connect_identity() as conn:
                with conn.cursor() as cur:
                    self._init_identity_schema(cur, same_database=False)
            self._bootstrap_dual_database_identity_storage()
            self._validate_dual_database_schema_state()
        else:
            with self._connect_biometric() as conn:
                with conn.cursor() as cur:
                    self._init_identity_schema(cur, same_database=True)
                    self._migrate_single_database_legacy_schema(cur)
                    self._harden_identity_schema(cur)
                    self._validate_single_database_schema_state(cur)

    @staticmethod
    def _require_vector_extension(cur) -> None:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_extension
                WHERE extname = 'vector'
            ) AS has_vector
            """
        )
        row = cur.fetchone()
        if not row or not bool(row['has_vector']):
            raise RuntimeError(
                'pgvector extension is not installed in biometric_db. '
                'Install it once with: CREATE EXTENSION vector;'
            )

    @staticmethod
    def _legacy_vector_check_constraint_sql() -> str:
        columns = tuple(sorted(LEGACY_VECTOR_COLUMN_BY_DIM.values()))
        clauses: List[str] = []
        for method in sorted(_legacy_storage_methods()):
            spec = VECTOR_SPECS[method]
            active_column = spec.legacy_storage_column
            if active_column is None:
                continue
            column_checks = [
                f"{column} IS {'NOT ' if column == active_column else ''}NULL"
                for column in columns
            ]
            clauses.append(
                "("
                + f"method = '{method}' AND dim = {int(spec.dim)} AND "
                + " AND ".join(column_checks)
                + ")"
            )
        return " OR ".join(clauses) or "FALSE"

    @staticmethod
    def _generic_vector_check_constraint_sql() -> str:
        columns = tuple(sorted(GENERIC_VECTOR_COLUMN_BY_DIM.values()))
        clauses: List[str] = []
        for dim, active_column in sorted(GENERIC_VECTOR_COLUMN_BY_DIM.items()):
            column_checks = [
                f"{column} IS {'NOT ' if column == active_column else ''}NULL"
                for column in columns
            ]
            clauses.append(
                "("
                + f"dim = {int(dim)} AND "
                + " AND ".join(column_checks)
                + ")"
            )
        return " OR ".join(clauses)

    def _init_biometric_schema(self, cur) -> None:
        self._require_vector_extension(cur)
        vector_check_sql = self._legacy_vector_check_constraint_sql()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.person_table} (
                random_id TEXT PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.idx_person_created_at}
            ON {self.person_table} (created_at DESC)
            """
        )

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.raw_table} (
                random_id TEXT PRIMARY KEY REFERENCES {self.person_table}(random_id) ON DELETE CASCADE,
                capture TEXT NOT NULL,
                ext TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                byte_size BIGINT,
                created_at TIMESTAMPTZ NOT NULL,
                image_bytes BYTEA
            )
            """
        )
        cur.execute(f"ALTER TABLE {self.raw_table} ADD COLUMN IF NOT EXISTS byte_size BIGINT")
        cur.execute(f"ALTER TABLE {self.raw_table} ALTER COLUMN image_bytes DROP NOT NULL")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.vector_table} (
                random_id TEXT NOT NULL REFERENCES {self.person_table}(random_id) ON DELETE CASCADE,
                method TEXT NOT NULL,
                dim INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                vector_512 vector(512),
                vector_768 vector(768),
                PRIMARY KEY (random_id, method),
                CHECK ({vector_check_sql})
            )
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.idx_vector_method_created_at}
            ON {self.vector_table} (method, created_at DESC)
            """
        )
        for method in sorted(_legacy_storage_methods()):
            spec = VECTOR_SPECS[method]
            if spec.legacy_storage_column is None:
                continue
            index_name = self.idx_vector_hnsw[method]
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {self.vector_table}
                USING hnsw ({spec.legacy_storage_column} vector_cosine_ops)
                WHERE method = '{method}'
                """
            )

        self._init_generic_vector_schema(cur)

    def _init_generic_vector_schema(self, cur) -> None:
        vector_check_sql = self._generic_vector_check_constraint_sql()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.generic_vector_table} (
                random_id TEXT NOT NULL REFERENCES {self.person_table}(random_id) ON DELETE CASCADE,
                method TEXT NOT NULL,
                vector_kind TEXT NOT NULL,
                dim INTEGER NOT NULL,
                distance_metric TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                vector_512 vector(512),
                vector_768 vector(768),
                vector_blob BYTEA,
                PRIMARY KEY (random_id, method, vector_kind),
                CHECK (BTRIM(method) <> ''),
                CHECK (BTRIM(vector_kind) <> ''),
                CHECK (BTRIM(distance_metric) <> ''),
                CHECK ({vector_check_sql})
            )
            """
        )
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ADD COLUMN IF NOT EXISTS random_id TEXT")
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ADD COLUMN IF NOT EXISTS method TEXT")
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ADD COLUMN IF NOT EXISTS vector_kind TEXT")
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ADD COLUMN IF NOT EXISTS dim INTEGER")
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ADD COLUMN IF NOT EXISTS distance_metric TEXT")
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ")
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ADD COLUMN IF NOT EXISTS vector_512 vector(512)")
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ADD COLUMN IF NOT EXISTS vector_768 vector(768)")
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ADD COLUMN IF NOT EXISTS metadata_json JSONB")
        cur.execute(
            f"""
            UPDATE {self.generic_vector_table}
            SET metadata_json = '{{}}'::jsonb
            WHERE metadata_json IS NULL
            """
        )
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ALTER COLUMN metadata_json SET DEFAULT '{{}}'::jsonb")
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ALTER COLUMN metadata_json SET NOT NULL")
        cur.execute(f"ALTER TABLE {self.generic_vector_table} ADD COLUMN IF NOT EXISTS vector_blob BYTEA")
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.idx_generic_vector_method_kind_created_at}
            ON {self.generic_vector_table} (method, vector_kind, created_at DESC)
            """
        )
        for dim, column in sorted(GENERIC_VECTOR_COLUMN_BY_DIM.items()):
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.idx_generic_vector_hnsw[dim]}
                ON {self.generic_vector_table}
                USING hnsw ({column} vector_cosine_ops)
                WHERE dim = {int(dim)}
                  AND distance_metric = 'cosine'
                  AND {column} IS NOT NULL
                """
            )
    def _init_identity_schema(self, cur, *, same_database: bool) -> None:
        if same_database:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.identity_table} (
                    random_id TEXT PRIMARY KEY REFERENCES {self.person_table}(random_id) ON DELETE CASCADE,
                    national_id TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL,
                    full_name TEXT,
                    name_norm TEXT
                )
                """
            )
        else:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.identity_table} (
                    random_id TEXT PRIMARY KEY,
                    national_id TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL,
                    full_name TEXT,
                    name_norm TEXT
                )
                """
            )
        cur.execute(f'ALTER TABLE {self.identity_table} ADD COLUMN IF NOT EXISTS full_name TEXT')
        cur.execute(f'ALTER TABLE {self.identity_table} ADD COLUMN IF NOT EXISTS name_norm TEXT')
        cur.execute(
            f'CREATE INDEX IF NOT EXISTS {self.idx_identity_national} ON {self.identity_table} (national_id text_pattern_ops)'
        )
        cur.execute(
            f'CREATE INDEX IF NOT EXISTS {self.idx_identity_name} ON {self.identity_table} (name_norm text_pattern_ops)'
        )
    def _migrate_single_database_legacy_schema(self, cur) -> None:
        legacy_rows = self._read_legacy_person_identity_rows(cur)
        current_rows = self._read_identity_rows_from_table(cur, self.identity_table)

        self._ensure_unique_identity_rows(
            legacy_rows,
            context=f'{self.person_table}.national_id',
        )
        self._ensure_unique_identity_rows(
            current_rows,
            context=self.identity_table,
        )

        if current_rows and legacy_rows and not self._mapping_rows_equal(current_rows, legacy_rows):
            raise RuntimeError(
                'Automatic migration aborted because legacy person_directory.national_id values conflict with the '
                f'existing {self.identity_table} rows in the same database.'
            )

        if not current_rows and legacy_rows:
            self._insert_identity_rows(cur, legacy_rows)
        elif legacy_rows:
            self._upsert_identity_rows(cur, legacy_rows)

        if self._column_exists(cur, self.person_table, 'national_id'):
            self._drop_legacy_person_national_id(cur)

        person_profile_rows = self._read_legacy_person_profile_rows(cur)
        if person_profile_rows:
            self._merge_person_profiles_into_identity(
                identity_cur=cur,
                person_rows=person_profile_rows,
                context_prefix=self.person_table,
            )

        missing_profiles = self._identity_rows_missing_profile(cur)
        if missing_profiles:
            raise RuntimeError(
                'Hardening migration aborted because identity_map rows are missing full_name/name_norm: '
                + ', '.join(missing_profiles)
            )
        if self._legacy_person_profile_columns_exist(cur):
            self._drop_legacy_person_profile_columns(cur)
    def _bootstrap_dual_database_identity_storage(self) -> None:
        with self._connect_biometric() as bio_conn, self._connect_identity() as ident_conn:
            with bio_conn.cursor() as bio_cur, ident_conn.cursor() as ident_cur:
                biometric_identity_rows = self._read_identity_rows_from_table(bio_cur, self.identity_table)
                legacy_person_rows = self._read_legacy_person_identity_rows(bio_cur)
                identity_rows = self._read_identity_rows_from_table(ident_cur, self.identity_table)

                self._ensure_unique_identity_rows(
                    biometric_identity_rows,
                    context=f'biometric_db.{self.identity_table}',
                )
                self._ensure_unique_identity_rows(
                    legacy_person_rows,
                    context=f'biometric_db.{self.person_table}.national_id',
                )
                self._ensure_unique_identity_rows(
                    identity_rows,
                    context=f'identity_db.{self.identity_table}',
                )

                if biometric_identity_rows and legacy_person_rows and not self._mapping_rows_equal(
                    biometric_identity_rows,
                    legacy_person_rows,
                ):
                    raise RuntimeError(
                        'Automatic migration aborted because biometric_db.identity_map and the legacy '
                        'person_directory.national_id column disagree. Resolve the mismatch and restart.'
                    )

                if not identity_rows:
                    source_rows = biometric_identity_rows or legacy_person_rows
                    if source_rows:
                        self._insert_identity_rows(ident_cur, source_rows)
                        copied_rows = self._read_identity_rows_from_table(ident_cur, self.identity_table)
                        if not self._mapping_rows_equal(source_rows, copied_rows):
                            raise RuntimeError(
                                'Automatic migration aborted because verification of copied rows into identity_db '
                                'failed. The legacy biometric mapping was left untouched.'
                            )
                else:
                    if biometric_identity_rows and not self._mapping_rows_equal(biometric_identity_rows, identity_rows):
                        raise RuntimeError(
                            'Automatic migration aborted because biometric_db.identity_map conflicts with the '
                            'existing identity_db.identity_map rows. Resolve the conflict and restart.'
                        )
                    if legacy_person_rows and not self._mapping_rows_equal(legacy_person_rows, identity_rows):
                        raise RuntimeError(
                            'Automatic migration aborted because biometric_db.person_directory.national_id conflicts '
                            'with the existing identity_db.identity_map rows. Resolve the conflict and restart.'
                        )

                person_profile_rows = self._read_legacy_person_profile_rows(bio_cur)
                if person_profile_rows:
                    self._merge_person_profiles_into_identity(
                        identity_cur=ident_cur,
                        person_rows=person_profile_rows,
                        context_prefix=f'biometric_db.{self.person_table}',
                    )

                missing_profiles = self._identity_rows_missing_profile(ident_cur)
                if missing_profiles:
                    raise RuntimeError(
                        'Hardening migration aborted because identity_db.identity_map rows are missing full_name/name_norm: '
                        + ', '.join(missing_profiles)
                    )

                self._harden_identity_schema(ident_cur)
                ident_conn.commit()

                if biometric_identity_rows or self._table_exists(bio_cur, self.identity_table):
                    self._drop_biometric_identity_table(bio_cur)
                if self._column_exists(bio_cur, self.person_table, 'national_id'):
                    self._drop_legacy_person_national_id(bio_cur)
                if self._legacy_person_profile_columns_exist(bio_cur):
                    self._drop_legacy_person_profile_columns(bio_cur)
                bio_conn.commit()

    @staticmethod
    def _identity_name_norm_sql_expression(full_name_column: str = 'full_name') -> str:
        return f"regexp_replace(lower(BTRIM({full_name_column})), '\\s+', ' ', 'g')"

    @staticmethod
    def _identity_national_id_sql_expression(column_name: str = 'national_id') -> str:
        return f"regexp_replace(COALESCE({column_name}, ''), '\\D', '', 'g')"

    def _harden_identity_schema(self, cur) -> None:
        self._normalize_identity_profile_rows(cur)
        self._normalize_identity_national_ids(cur)

        missing_profiles = self._identity_rows_missing_profile(cur)
        if missing_profiles:
            raise RuntimeError(
                'Hardening migration aborted because identity_map rows are missing full_name/name_norm: '
                + ', '.join(missing_profiles)
            )

        noncanonical_name_norm = self._identity_rows_with_noncanonical_name_norm(cur)
        if noncanonical_name_norm:
            raise RuntimeError(
                'Hardening migration aborted because identity_map.name_norm does not match the normalized full_name '
                'contract for random_id values: '
                + ', '.join(noncanonical_name_norm)
            )

        invalid_national_ids = self._identity_rows_with_invalid_national_id(cur)
        if invalid_national_ids:
            raise RuntimeError(
                'Hardening migration aborted because identity_map.national_id is not a non-empty digits-only value '
                'for random_id values: '
                + ', '.join(invalid_national_ids)
            )

        self._ensure_column_not_null(cur, table_name=self.identity_table, column_name='full_name')
        self._ensure_column_not_null(cur, table_name=self.identity_table, column_name='name_norm')
        self._ensure_column_not_null(cur, table_name=self.identity_table, column_name='national_id')

        self._ensure_check_constraint(
            cur,
            table_name=self.identity_table,
            constraint_name=self.ck_identity_full_name_not_blank,
            check_sql="BTRIM(full_name) <> ''",
        )
        self._ensure_check_constraint(
            cur,
            table_name=self.identity_table,
            constraint_name=self.ck_identity_name_norm_not_blank,
            check_sql="BTRIM(name_norm) <> ''",
        )
        self._ensure_check_constraint(
            cur,
            table_name=self.identity_table,
            constraint_name=self.ck_identity_name_norm_matches_full_name,
            check_sql=f"name_norm = {self._identity_name_norm_sql_expression()}",
        )
        self._ensure_check_constraint(
            cur,
            table_name=self.identity_table,
            constraint_name=self.ck_identity_national_id_digits_only,
            check_sql="national_id ~ '^[0-9]+$'",
        )

        missing_contract_items = self._missing_identity_contract_items(cur)
        if missing_contract_items:
            raise RuntimeError(
                'Schema hardening failed because identity_map is still missing enforced contract items: '
                + ', '.join(missing_contract_items)
            )

    def _normalize_identity_profile_rows(self, cur) -> None:
        name_norm_expr = self._identity_name_norm_sql_expression()
        cur.execute(
            f"""
            UPDATE {self.identity_table}
            SET full_name = BTRIM(full_name)
            WHERE full_name IS NOT NULL
              AND full_name <> BTRIM(full_name)
            """
        )
        cur.execute(
            f"""
            UPDATE {self.identity_table}
            SET name_norm = {name_norm_expr}
            WHERE full_name IS NOT NULL
              AND BTRIM(full_name) <> ''
              AND (
                  name_norm IS NULL
                  OR BTRIM(name_norm) = ''
                  OR name_norm <> {name_norm_expr}
              )
            """
        )

    def _normalize_identity_national_ids(self, cur) -> None:
        normalized_expr = self._identity_national_id_sql_expression()
        blank_rows = self._sample_identity_random_ids(cur, where_sql=f"{normalized_expr} = ''")
        if blank_rows:
            raise RuntimeError(
                'Hardening migration aborted because identity_map.national_id is blank after normalization for '
                'random_id values: '
                + ', '.join(blank_rows)
            )

        duplicate_rows = self._sample_duplicate_normalized_identity_national_ids(cur)
        if duplicate_rows:
            duplicates_text = '; '.join(
                f"{national_id} ({', '.join(random_ids)})"
                for national_id, random_ids in duplicate_rows
            )
            raise RuntimeError(
                'Hardening migration aborted because identity_map.national_id values collapse to duplicate '
                'normalized identifiers: '
                + duplicates_text
            )

        cur.execute(
            f"""
            UPDATE {self.identity_table}
            SET national_id = {normalized_expr}
            WHERE national_id IS DISTINCT FROM {normalized_expr}
            """
        )

    def _sample_identity_random_ids(self, cur, *, where_sql: str, limit: int = 10) -> List[str]:
        cur.execute(
            f"""
            SELECT random_id
            FROM {self.identity_table}
            WHERE {where_sql}
            ORDER BY random_id
            LIMIT %s
            """,
            (int(limit),),
        )
        return [str(row['random_id']) for row in cur.fetchall()]

    def _sample_duplicate_normalized_identity_national_ids(
        self,
        cur,
        *,
        limit: int = 5,
    ) -> List[tuple[str, List[str]]]:
        normalized_expr = self._identity_national_id_sql_expression()
        cur.execute(
            f"""
            SELECT national_id_norm, ARRAY_AGG(random_id ORDER BY random_id) AS random_ids
            FROM (
                SELECT random_id, {normalized_expr} AS national_id_norm
                FROM {self.identity_table}
            ) AS normalized_rows
            WHERE national_id_norm <> ''
            GROUP BY national_id_norm
            HAVING COUNT(*) > 1
            ORDER BY national_id_norm
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        return [
            (
                str(row['national_id_norm']),
                [str(random_id) for random_id in row['random_ids']],
            )
            for row in rows
        ]

    def _identity_rows_with_noncanonical_name_norm(self, cur, *, limit: int = 10) -> List[str]:
        return self._sample_identity_random_ids(
            cur,
            where_sql=(
                "full_name IS NOT NULL "
                "AND BTRIM(full_name) <> '' "
                f"AND name_norm <> {self._identity_name_norm_sql_expression()}"
            ),
            limit=limit,
        )

    def _identity_rows_with_invalid_national_id(self, cur, *, limit: int = 10) -> List[str]:
        return self._sample_identity_random_ids(
            cur,
            where_sql="national_id IS NULL OR BTRIM(national_id) = '' OR national_id !~ '^[0-9]+$'",
            limit=limit,
        )

    def _format_dual_database_operation_error(
        self,
        *,
        operation: str,
        commit_state: str,
        manual_reconciliation_required: bool,
        exc: Exception,
        compensation_status: str | None = None,
    ) -> str:
        parts = [
            f"Dual-database {operation} failed.",
            f"Commit state: {commit_state}.",
        ]
        if compensation_status:
            parts.append(f"Compensation: {compensation_status}.")
        parts.append(
            "Manual reconciliation required: "
            + ("yes" if manual_reconciliation_required else "no")
            + "."
        )
        parts.append(
            "Next step: run "
            + self._reconciliation_script_command()
            + "."
        )
        parts.append(f"Details: {exc}")
        return " ".join(parts)

    def _validate_single_database_schema_state(self, cur) -> None:
        if self._column_exists(cur, self.person_table, 'national_id'):
            raise RuntimeError(
                f'Schema validation failed: {self.person_table}.national_id still exists after migration.'
            )
        if self._legacy_person_profile_columns_exist(cur):
            raise RuntimeError(
                f'Schema validation failed: {self.person_table} still contains legacy full_name/name_norm columns.'
            )

        cur.execute(
            f"""
            SELECT p.random_id
            FROM {self.person_table} AS p
            LEFT JOIN {self.identity_table} AS i ON i.random_id = p.random_id
            WHERE i.random_id IS NULL
            ORDER BY p.random_id
            LIMIT 10
            """
        )
        missing_rows = cur.fetchall()
        if missing_rows:
            sample = ', '.join(str(row['random_id']) for row in missing_rows)
            raise RuntimeError(
                'Schema validation failed: person_directory contains rows without matching identity_map entries: '
                f'{sample}'
            )

        cur.execute(
            f"""
            SELECT i.random_id
            FROM {self.identity_table} AS i
            LEFT JOIN {self.person_table} AS p ON p.random_id = i.random_id
            WHERE p.random_id IS NULL
            ORDER BY i.random_id
            LIMIT 10
            """
        )
        orphan_rows = cur.fetchall()
        if orphan_rows:
            sample = ', '.join(str(row['random_id']) for row in orphan_rows)
            raise RuntimeError(
                'Schema validation failed: identity_map contains orphan mapping rows: '
                f'{sample}'
            )

        missing_profiles = self._identity_rows_missing_profile(cur)
        if missing_profiles:
            raise RuntimeError(
                'Schema validation failed: identity_map rows are missing full_name/name_norm: '
                + ', '.join(missing_profiles)
            )
        missing_contract_items = self._missing_identity_contract_items(cur)
        if missing_contract_items:
            raise RuntimeError(
                'Schema validation failed: identity_map is missing hardened contract items: '
                + ', '.join(missing_contract_items)
            )
    def _validate_dual_database_schema_state(self) -> None:
        try:
            with self._connect_biometric() as bio_conn:
                with bio_conn.cursor() as bio_cur:
                    if self._column_exists(bio_cur, self.person_table, 'national_id'):
                        raise RuntimeError(
                            f'Schema validation failed: {self.person_table}.national_id still exists in biometric_db.'
                        )
                    if self._table_exists(bio_cur, self.identity_table):
                        raise RuntimeError(
                            f'Schema validation failed: legacy biometric_db table {self.identity_table} still exists.'
                        )
                    if self._legacy_person_profile_columns_exist(bio_cur):
                        raise RuntimeError(
                            f'Schema validation failed: {self.person_table} still contains legacy full_name/name_norm columns.'
                        )
                    people_count = self._count_rows(bio_cur, self.person_table)

            with self._connect_identity() as ident_conn:
                with ident_conn.cursor() as ident_cur:
                    identity_count = self._count_rows(ident_cur, self.identity_table)
                    missing_profiles = self._identity_rows_missing_profile(ident_cur)
                    missing_contract_items = self._missing_identity_contract_items(ident_cur)

            drift = self._inspect_dual_database_identity_drift(
                people_count=int(people_count),
                identity_count=int(identity_count),
                sample_limit=RECONCILIATION_SAMPLE_LIMIT,
            )
            if drift.people_without_identity_sample:
                raise RuntimeError(
                    'Schema validation failed: biometric_db contains people without matching identity_db mappings: '
                    + ', '.join(drift.people_without_identity_sample)
                )
            if drift.identity_without_people_sample:
                raise RuntimeError(
                    'Schema validation failed: identity_db contains orphan mapping rows: '
                    + ', '.join(drift.identity_without_people_sample)
                )

            if missing_profiles:
                raise RuntimeError(
                    'Schema validation failed: identity_db.identity_map rows are missing full_name/name_norm: '
                    + ', '.join(missing_profiles)
                )
            if missing_contract_items:
                raise RuntimeError(
                    'Schema validation failed: identity_db.identity_map is missing hardened contract items: '
                    + ', '.join(missing_contract_items)
                )
        except Exception as exc:
            raise RuntimeError(
                self._format_dual_database_operation_error(
                    operation="startup_validation",
                    commit_state="validation/read-only check; no side committed during the failing step",
                    manual_reconciliation_required=True,
                    exc=exc,
                )
            ) from exc
    def _enroll_dual_database(
        self,
        *,
        full_name: str,
        name_norm: str,
        national_id_norm: str,
        capture_norm: str,
        ext: str,
        vector_payload: Dict[str, np.ndarray],
        random_id: str,
        created_at_dt: datetime,
        created_at_iso: str,
        image_hash: str,
        byte_size: int | None,
        replace_existing: bool,
    ) -> EnrollmentReceipt:
        bio_conn = self._connect_biometric()
        ident_conn = self._connect_identity()
        identity_commit_done = False
        previous_mapping: Optional[_IdentityMapRow] = None
        receipt: EnrollmentReceipt | None = None

        try:
            with bio_conn.cursor() as bio_cur, ident_conn.cursor() as ident_cur:
                previous_mapping = self._lookup_identity_row_by_national_id_with_cursor(ident_cur, national_id_norm)
                if previous_mapping is not None and not replace_existing:
                    raise ValueError(
                        'national_id already enrolled; pass replace_existing=True to rotate the template'
                    )

                if previous_mapping is not None:
                    bio_cur.execute(
                        f'DELETE FROM {self.person_table} WHERE random_id = %s',
                        (previous_mapping.random_id,),
                    )

                self._insert_person_row(
                    bio_cur,
                    random_id=random_id,
                    created_at=created_at_dt,
                )
                self._insert_raw_row(
                    bio_cur,
                    random_id=random_id,
                    capture=capture_norm,
                    ext=ext,
                    sha256=image_hash,
                    byte_size=byte_size,
                    created_at=created_at_dt,
                )
                self._insert_vector_rows(
                    bio_cur,
                    random_id=random_id,
                    created_at=created_at_dt,
                    vector_payload=vector_payload,
                )

                if previous_mapping is not None:
                    ident_cur.execute(
                        f'DELETE FROM {self.identity_table} WHERE random_id = %s',
                        (previous_mapping.random_id,),
                    )

                self._insert_identity_rows(
                    ident_cur,
                    [
                        _IdentityMapRow(
                            random_id=random_id,
                            national_id=national_id_norm,
                            created_at=created_at_dt,
                            full_name=full_name,
                            name_norm=name_norm,
                        )
                    ],
                )
                ident_conn.commit()
                identity_commit_done = True
                bio_conn.commit()

                receipt = EnrollmentReceipt(
                    random_id=random_id,
                    created_at=created_at_iso,
                    vector_methods=sorted(vector_payload.keys()),
                    image_sha256=image_hash,
                )
        except Exception as exc:
            self._safe_rollback(bio_conn)
            if isinstance(exc, ValueError):
                self._safe_rollback(ident_conn)
                raise
            if identity_commit_done:
                try:
                    self._compensate_enroll_identity_failure(
                        new_random_id=random_id,
                        previous_mapping=previous_mapping,
                    )
                except Exception as comp_exc:
                    raise RuntimeError(
                        self._format_dual_database_operation_error(
                            operation="enroll",
                            commit_state="after identity_db committed; biometric_db had not committed successfully",
                            manual_reconciliation_required=True,
                            exc=exc,
                            compensation_status=f"compensating rollback failed ({comp_exc})",
                        )
                    ) from exc
                raise RuntimeError(
                    self._format_dual_database_operation_error(
                        operation="enroll",
                        commit_state="after identity_db committed; biometric_db had not committed successfully",
                        manual_reconciliation_required=False,
                        exc=exc,
                        compensation_status="compensating rollback completed successfully",
                    )
                ) from exc
            else:
                self._safe_rollback(ident_conn)
                raise RuntimeError(
                    self._format_dual_database_operation_error(
                        operation="enroll",
                        commit_state="before either side committed",
                        manual_reconciliation_required=False,
                        exc=exc,
                        compensation_status="both transactions rolled back",
                    )
                ) from exc
        finally:
            ident_conn.close()
            bio_conn.close()

        if receipt is None:
            raise RuntimeError('Enrollment completed without producing a receipt')
        return receipt
    def _purge_dual_database(self, random_id: str) -> bool:
        bio_conn = self._connect_biometric()
        ident_conn = self._connect_identity()
        identity_commit_done = False
        previous_mapping: Optional[_IdentityMapRow] = None
        deleted_any = False

        try:
            with bio_conn.cursor() as bio_cur, ident_conn.cursor() as ident_cur:
                previous_mapping = self._lookup_identity_row_by_random_id_with_cursor(ident_cur, random_id)

                bio_cur.execute(
                    f'DELETE FROM {self.person_table} WHERE random_id = %s',
                    (random_id,),
                )
                deleted_any = deleted_any or (bio_cur.rowcount > 0)

                ident_cur.execute(
                    f'DELETE FROM {self.identity_table} WHERE random_id = %s',
                    (random_id,),
                )
                deleted_any = deleted_any or (ident_cur.rowcount > 0)
                ident_conn.commit()
                identity_commit_done = True
                bio_conn.commit()
        except Exception as exc:
            self._safe_rollback(bio_conn)
            if identity_commit_done and previous_mapping is not None:
                try:
                    self._restore_identity_row(previous_mapping)
                except Exception as comp_exc:
                    raise RuntimeError(
                        self._format_dual_database_operation_error(
                            operation="purge",
                            commit_state="after identity_db committed; biometric_db had not committed successfully",
                            manual_reconciliation_required=True,
                            exc=exc,
                            compensation_status=f"identity_db restore failed ({comp_exc})",
                        )
                    ) from exc
                raise RuntimeError(
                    self._format_dual_database_operation_error(
                        operation="purge",
                        commit_state="after identity_db committed; biometric_db had not committed successfully",
                        manual_reconciliation_required=False,
                        exc=exc,
                        compensation_status="identity_db restore completed successfully",
                    )
                ) from exc

            self._safe_rollback(ident_conn)
            raise RuntimeError(
                self._format_dual_database_operation_error(
                    operation="purge",
                    commit_state="before either side committed"
                    if not identity_commit_done
                    else "after identity_db committed with no identity row change requiring restore",
                    manual_reconciliation_required=False,
                    exc=exc,
                    compensation_status="both transactions rolled back"
                    if not identity_commit_done
                    else "no identity_db restore was required",
                )
            ) from exc
        finally:
            ident_conn.close()
            bio_conn.close()

        return deleted_any
    def _enroll_single_database(
        self,
        *,
        full_name: str,
        name_norm: str,
        national_id_norm: str,
        capture_norm: str,
        ext: str,
        vector_payload: Dict[str, np.ndarray],
        random_id: str,
        created_at_dt: datetime,
        created_at_iso: str,
        image_hash: str,
        byte_size: int | None,
        replace_existing: bool,
    ) -> EnrollmentReceipt:
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                existing_random_id = self._lookup_random_id_with_cursor(cur, national_id_norm)
                if existing_random_id is not None and not replace_existing:
                    raise ValueError(
                        'national_id already enrolled; pass replace_existing=True to rotate the template'
                    )
                if existing_random_id is not None and replace_existing:
                    cur.execute(f'DELETE FROM {self.person_table} WHERE random_id = %s', (existing_random_id,))

                self._insert_person_row(
                    cur,
                    random_id=random_id,
                    created_at=created_at_dt,
                )
                self._insert_identity_rows(
                    cur,
                    [
                        _IdentityMapRow(
                            random_id=random_id,
                            national_id=national_id_norm,
                            created_at=created_at_dt,
                            full_name=full_name,
                            name_norm=name_norm,
                        )
                    ],
                )
                self._insert_raw_row(
                    cur,
                    random_id=random_id,
                    capture=capture_norm,
                    ext=ext,
                    sha256=image_hash,
                    byte_size=byte_size,
                    created_at=created_at_dt,
                )
                self._insert_vector_rows(
                    cur,
                    random_id=random_id,
                    created_at=created_at_dt,
                    vector_payload=vector_payload,
                )

        return EnrollmentReceipt(
            random_id=random_id,
            created_at=created_at_iso,
            vector_methods=sorted(vector_payload.keys()),
            image_sha256=image_hash,
        )
    def _purge_single_database(self, random_id: str) -> bool:
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.identity_table} WHERE random_id = %s", (random_id,))
                cur.execute(f"DELETE FROM {self.person_table} WHERE random_id = %s", (random_id,))
                return bool(cur.rowcount)

    # ------------------------------------------------------------------
    # read helpers
    # ------------------------------------------------------------------
    def _search_people_single_database(self, hints: IdentifyHints, *, limit: int | None = None) -> List[PersonDirectoryRecord]:
        sql = [
            f"""
            SELECT
                p.random_id,
                COALESCE(i.full_name, '') AS full_name,
                COALESCE(i.name_norm, '') AS name_norm,
                COALESCE(i.national_id, '') AS national_id,
                p.created_at
            FROM {self.person_table} AS p
            LEFT JOIN {self.identity_table} AS i ON i.random_id = p.random_id
            """
        ]
        clauses: List[str] = []
        params: List[object] = []

        if hints.name_pattern:
            clauses.append('i.name_norm LIKE %s')
            params.append(self._pattern_to_like(normalize_name(hints.name_pattern)))

        if hints.national_id_pattern:
            raw = normalize_national_id(hints.national_id_pattern) or hints.national_id_pattern
            clauses.append('i.national_id LIKE %s')
            params.append(self._pattern_to_like(str(raw)))

        self._append_created_at_filters(
            clauses,
            params,
            created_from=hints.created_from,
            created_to=hints.created_to,
            column_name='p.created_at',
        )

        if clauses:
            sql.append('WHERE ' + ' AND '.join(clauses))
        sql.append('ORDER BY p.created_at DESC')
        if limit is not None:
            sql.append('LIMIT %s')
            params.append(int(limit))

        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(' '.join(sql), params)
                rows = cur.fetchall()
        return [self._row_to_person_record(row) for row in rows]
    def _get_person_single_database(self, random_id: str) -> Optional[PersonDirectoryRecord]:
        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        p.random_id,
                        COALESCE(i.full_name, '') AS full_name,
                        COALESCE(i.name_norm, '') AS name_norm,
                        COALESCE(i.national_id, '') AS national_id,
                        p.created_at
                    FROM {self.person_table} AS p
                    LEFT JOIN {self.identity_table} AS i ON i.random_id = p.random_id
                    WHERE p.random_id = %s
                    """,
                    (random_id,),
                )
                row = cur.fetchone()
        return self._row_to_person_record(row) if row else None
    def _query_people_rows(
        self,
        *,
        created_from: str | None,
        created_to: str | None,
        candidate_ids: Sequence[str] | None,
        limit: int | None,
    ):
        sql = [f'SELECT random_id, created_at FROM {self.person_table}']
        clauses: List[str] = []
        params: List[object] = []

        if candidate_ids is not None:
            if not candidate_ids:
                return []
            clauses.append('random_id = ANY(%s)')
            params.append(list(candidate_ids))

        self._append_created_at_filters(
            clauses,
            params,
            created_from=created_from,
            created_to=created_to,
        )

        if clauses:
            sql.append('WHERE ' + ' AND '.join(clauses))
        sql.append('ORDER BY created_at DESC, random_id')
        if limit is not None:
            sql.append('LIMIT %s')
            params.append(int(limit))

        with self._connect_biometric() as conn:
            with conn.cursor() as cur:
                cur.execute(' '.join(sql), params)
                return cur.fetchall()
    def _query_identity_rows(
        self,
        *,
        name_pattern: str | None,
        national_id_pattern: str | None,
        candidate_ids: Sequence[str] | None,
        limit: int | None,
    ) -> List[_IdentityMapRow]:
        sql = [
            f"""
            SELECT random_id, national_id, created_at, full_name, name_norm
            FROM {self.identity_table}
            """
        ]
        clauses: List[str] = []
        params: List[object] = []

        if candidate_ids is not None:
            if not candidate_ids:
                return []
            clauses.append('random_id = ANY(%s)')
            params.append(list(candidate_ids))

        if name_pattern:
            clauses.append('name_norm LIKE %s')
            params.append(self._pattern_to_like(normalize_name(name_pattern)))

        if national_id_pattern:
            raw = normalize_national_id(national_id_pattern) or national_id_pattern
            clauses.append('national_id LIKE %s')
            params.append(self._pattern_to_like(str(raw)))

        if clauses:
            sql.append('WHERE ' + ' AND '.join(clauses))
        sql.append('ORDER BY created_at DESC, random_id')
        if limit is not None:
            sql.append('LIMIT %s')
            params.append(int(limit))

        with self._connect_identity() as conn:
            with conn.cursor() as cur:
                cur.execute(' '.join(sql), params)
                rows = cur.fetchall()
        return [self._row_to_identity_row(row) for row in rows]

    def _load_identity_rows_by_random_ids(self, random_ids: Sequence[str]) -> List[_IdentityMapRow]:
        return self._query_identity_rows(
            name_pattern=None,
            national_id_pattern=None,
            candidate_ids=random_ids,
            limit=None,
        )

    def _lookup_identity_row_by_random_id(self, random_id: str) -> Optional[_IdentityMapRow]:
        with self._connect_identity() as conn:
            with conn.cursor() as cur:
                return self._lookup_identity_row_by_random_id_with_cursor(cur, random_id)

    def _lookup_identity_row_by_random_id_with_cursor(self, cur, random_id: str) -> Optional[_IdentityMapRow]:
        cur.execute(
            f"""
            SELECT random_id, national_id, created_at, full_name, name_norm
            FROM {self.identity_table}
            WHERE random_id = %s
            """,
            (random_id,),
        )
        row = cur.fetchone()
        return self._row_to_identity_row(row) if row else None

    def _lookup_identity_row_by_national_id_with_cursor(self, cur, national_id_norm: str) -> Optional[_IdentityMapRow]:
        cur.execute(
            f"""
            SELECT random_id, national_id, created_at, full_name, name_norm
            FROM {self.identity_table}
            WHERE national_id = %s
            """,
            (national_id_norm,),
        )
        row = cur.fetchone()
        return self._row_to_identity_row(row) if row else None

    # ------------------------------------------------------------------
    # SQL helpers
    # ------------------------------------------------------------------
    def _insert_person_row(self, cur, *, random_id: str, created_at: datetime) -> None:
        cur.execute(
            f"""
            INSERT INTO {self.person_table} (random_id, created_at)
            VALUES (%s, %s)
            """,
            (random_id, created_at),
        )
    def _insert_raw_row(
        self,
        cur,
        *,
        random_id: str,
        capture: str,
        ext: str,
        sha256: str,
        byte_size: int | None,
        created_at: datetime,
    ) -> None:
        cur.execute(
            f"""
            INSERT INTO {self.raw_table} (random_id, capture, ext, sha256, byte_size, created_at, image_bytes)
            VALUES (%s, %s, %s, %s, %s, %s, NULL)
            """,
            (random_id, capture, ext, sha256, byte_size, created_at),
        )

    @staticmethod
    def _vector_payloads_for_column(
        vec: np.ndarray,
        *,
        column: str,
        method: str,
    ) -> tuple[list[float] | None, list[float] | None]:
        if column not in GENERIC_VECTOR_COLUMN_BY_DIM.values():  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported vector column {column!r} for method={method}")
        payload = vec.tolist()
        return (
            payload if column == "vector_512" else None,
            payload if column == "vector_768" else None,
        )

    def _insert_vector_rows(
        self,
        cur,
        *,
        random_id: str,
        created_at: datetime,
        vector_payload: Dict[str, np.ndarray],
    ) -> None:
        for method, vec in vector_payload.items():
            spec = self._spec_for_method(method)
            if spec.generic_storage_enabled:
                vector_512, vector_768 = self._vector_payloads_for_column(
                    vec,
                    column=spec.generic_storage_column,
                    method=spec.method,
                )
                cur.execute(
                    f"""
                    INSERT INTO {self.generic_vector_table}
                        (
                            random_id,
                            method,
                            vector_kind,
                            dim,
                            distance_metric,
                            created_at,
                            metadata_json,
                            vector_512,
                            vector_768
                        )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (random_id, method, vector_kind)
                    DO UPDATE SET
                        dim = EXCLUDED.dim,
                        distance_metric = EXCLUDED.distance_metric,
                        created_at = EXCLUDED.created_at,
                        metadata_json = EXCLUDED.metadata_json,
                        vector_512 = EXCLUDED.vector_512,
                        vector_768 = EXCLUDED.vector_768
                    """,
                    (
                        random_id,
                        spec.method,
                        spec.vector_kind,
                        spec.dim,
                        spec.distance_metric,
                        created_at,
                        json.dumps({"source": "enroll"}),
                        vector_512,
                        vector_768,
                    ),
                )

            if spec.legacy_storage_enabled:
                if spec.legacy_storage_column is None:  # pragma: no cover - defensive branch
                    raise ValueError(f"Legacy storage enabled without a legacy column for method={method}")
                vector_512, vector_768 = self._vector_payloads_for_column(
                    vec,
                    column=spec.legacy_storage_column,
                    method=spec.method,
                )
                cur.execute(
                    f"""
                    INSERT INTO {self.vector_table}
                        (random_id, method, dim, created_at, vector_512, vector_768)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (random_id, spec.method, spec.dim, created_at, vector_512, vector_768),
                )

    def _insert_identity_rows(self, cur, rows: Sequence[_IdentityMapRow]) -> None:
        if not rows:
            return
        cur.executemany(
            f"""
            INSERT INTO {self.identity_table} (random_id, national_id, created_at, full_name, name_norm)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [(row.random_id, row.national_id, row.created_at, row.full_name, row.name_norm) for row in rows],
        )
    def _drop_biometric_identity_table(self, cur) -> None:
        cur.execute(f"DROP TABLE IF EXISTS {self.identity_table}")

    def _drop_legacy_person_national_id(self, cur) -> None:
        cur.execute(f"DROP INDEX IF EXISTS {self.legacy_idx_person_national}")
        cur.execute(f"ALTER TABLE {self.person_table} DROP COLUMN IF EXISTS national_id CASCADE")


    def _upsert_identity_rows(self, cur, rows: Sequence[_IdentityMapRow]) -> None:
        if not rows:
            return
        cur.executemany(
            f"""
            INSERT INTO {self.identity_table} (random_id, national_id, created_at, full_name, name_norm)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (random_id)
            DO UPDATE SET
                national_id = EXCLUDED.national_id,
                created_at = EXCLUDED.created_at,
                full_name = EXCLUDED.full_name,
                name_norm = EXCLUDED.name_norm
            """,
            [(row.random_id, row.national_id, row.created_at, row.full_name, row.name_norm) for row in rows],
        )

    def _legacy_person_profile_columns_exist(self, cur) -> bool:
        has_full_name = self._column_exists(cur, self.person_table, 'full_name')
        has_name_norm = self._column_exists(cur, self.person_table, 'name_norm')
        if has_full_name != has_name_norm:
            raise RuntimeError(
                f'Schema validation failed: {self.person_table} has only one of full_name/name_norm.'
            )
        return has_full_name and has_name_norm

    def _read_legacy_person_profile_rows(self, cur) -> List[_LegacyPersonProfileRow]:
        if not self._legacy_person_profile_columns_exist(cur):
            return []
        cur.execute(
            f"""
            SELECT random_id, full_name, name_norm, created_at
            FROM {self.person_table}
            ORDER BY random_id
            """
        )
        rows = cur.fetchall()
        result: List[_LegacyPersonProfileRow] = []
        for row in rows:
            full_name = '' if row['full_name'] is None else str(row['full_name']).strip()
            if not full_name:
                raise RuntimeError(
                    f'Hardening migration aborted because {self.person_table}.full_name is blank for random_id={row["random_id"]}.'
                )
            raw_name_norm = '' if row['name_norm'] is None else str(row['name_norm']).strip()
            name_norm = raw_name_norm or normalize_name(full_name)
            if not name_norm:
                raise RuntimeError(
                    f'Hardening migration aborted because {self.person_table}.name_norm is blank for random_id={row["random_id"]}.'
                )
            result.append(
                _LegacyPersonProfileRow(
                    random_id=str(row['random_id']),
                    full_name=full_name,
                    name_norm=name_norm,
                    created_at=self._as_utc_datetime(row['created_at']),
                )
            )
        return result

    def _merge_person_profiles_into_identity(
        self,
        *,
        identity_cur,
        person_rows: Sequence[_LegacyPersonProfileRow],
        context_prefix: str,
    ) -> None:
        pending: List[_IdentityMapRow] = []
        for person_row in person_rows:
            current = self._lookup_identity_row_by_random_id_with_cursor(identity_cur, person_row.random_id)
            if current is None:
                raise RuntimeError(
                    f'Hardening migration aborted because {context_prefix} contains rows without matching identity_map rows.'
                )
            if current.full_name and current.full_name != person_row.full_name:
                raise RuntimeError(
                    f'Hardening migration aborted because identity_map.full_name disagrees for random_id={person_row.random_id}.'
                )
            if current.name_norm and current.name_norm != person_row.name_norm:
                raise RuntimeError(
                    f'Hardening migration aborted because identity_map.name_norm disagrees for random_id={person_row.random_id}.'
                )
            pending.append(
                _IdentityMapRow(
                    random_id=current.random_id,
                    national_id=current.national_id,
                    created_at=current.created_at,
                    full_name=person_row.full_name,
                    name_norm=person_row.name_norm,
                )
            )
        self._upsert_identity_rows(identity_cur, pending)

    def _identity_rows_missing_profile(self, cur) -> List[str]:
        cur.execute(
            f"""
            SELECT random_id
            FROM {self.identity_table}
            WHERE full_name IS NULL OR BTRIM(full_name) = ''
               OR name_norm IS NULL OR BTRIM(name_norm) = ''
            ORDER BY random_id
            LIMIT 10
            """
        )
        return [str(row['random_id']) for row in cur.fetchall()]

    def _drop_legacy_person_profile_columns(self, cur) -> None:
        cur.execute(f'DROP INDEX IF EXISTS {self.idx_person_name}')
        cur.execute(f'ALTER TABLE {self.person_table} DROP COLUMN IF EXISTS full_name CASCADE')
        cur.execute(f'ALTER TABLE {self.person_table} DROP COLUMN IF EXISTS name_norm CASCADE')

    def _append_created_at_filters(
        self,
        clauses: List[str],
        params: List[object],
        *,
        created_from: str | None,
        created_to: str | None,
        column_name: str = 'created_at',
    ) -> None:
        if created_from:
            clauses.append(f'{column_name} >= %s')
            params.append(self._normalize_date_lower(created_from))
        if created_to:
            clauses.append(f'{column_name} <= %s')
            params.append(self._normalize_date_upper(created_to))

    # ------------------------------------------------------------------
    # compensation helpers
    # ------------------------------------------------------------------
    def _compensate_enroll_identity_failure(
        self,
        *,
        new_random_id: str,
        previous_mapping: Optional[_IdentityMapRow],
    ) -> None:
        with self._connect_identity() as conn:
            with conn.cursor() as cur:
                cur.execute(f'DELETE FROM {self.identity_table} WHERE random_id = %s', (new_random_id,))
                if previous_mapping is not None:
                    cur.execute(
                        f"""
                        INSERT INTO {self.identity_table} (random_id, national_id, created_at, full_name, name_norm)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (random_id)
                        DO UPDATE SET
                            national_id = EXCLUDED.national_id,
                            created_at = EXCLUDED.created_at,
                            full_name = EXCLUDED.full_name,
                            name_norm = EXCLUDED.name_norm
                        """,
                        (
                            previous_mapping.random_id,
                            previous_mapping.national_id,
                            previous_mapping.created_at,
                            previous_mapping.full_name,
                            previous_mapping.name_norm,
                        ),
                    )
    def _restore_identity_row(self, row: _IdentityMapRow) -> None:
        with self._connect_identity() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.identity_table} (random_id, national_id, created_at, full_name, name_norm)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (random_id)
                    DO UPDATE SET
                        national_id = EXCLUDED.national_id,
                        created_at = EXCLUDED.created_at,
                        full_name = EXCLUDED.full_name,
                        name_norm = EXCLUDED.name_norm
                    """,
                    (row.random_id, row.national_id, row.created_at, row.full_name, row.name_norm),
                )
    def _connect_biometric(self, *, autocommit: bool = False):
        return self._connect_postgres(
            self.biometric_database_url,
            autocommit=autocommit,
            database_role="biometric",
            needs_vector=True,
        )

    def _connect_identity(self, *, autocommit: bool = False):
        return self._connect_postgres(
            self.identity_database_url,
            autocommit=autocommit,
            database_role="identity",
            needs_vector=False,
        )

    def _connect_postgres(
        self,
        database_url: str,
        *,
        autocommit: bool,
        database_role: str,
        needs_vector: bool,
    ):
        psycopg, dict_row = _load_postgres_base_deps()
        try:
            conn = psycopg.connect(database_url, autocommit=autocommit, row_factory=dict_row)
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                f'Failed to connect to {database_role} database ({self._redacted_database_url(database_url)}): {exc}'
            ) from exc

        try:
            if needs_vector:
                register_vector = _load_pgvector_register()
                try:
                    register_vector(conn)
                except Exception as exc:
                    if not _looks_like_missing_vector_type_error(exc):
                        raise
                    try:
                        with conn.cursor() as cur:
                            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                        if not autocommit:
                            conn.commit()
                        register_vector(conn)
                    except Exception as create_exc:
                        raise RuntimeError(
                            f"pgvector extension is not available in the {database_role} database "
                            f"({self._redacted_database_url(database_url)}). "
                            "Install or enable it with CREATE EXTENSION vector."
                        ) from create_exc
            return conn
        except Exception:
            conn.close()
            raise

    def _schema_index_contract(self) -> List[Dict[str, str]]:
        base = [
            {
                "key": "person_created_at_desc",
                "database_role": "biometric_db",
                "table_key": "person",
                "name": self.idx_person_created_at,
            },
            {
                "key": "identity_name_norm_prefix",
                "database_role": "identity_db",
                "table_key": "identity",
                "name": self.idx_identity_name,
            },
            {
                "key": "identity_national_id_prefix",
                "database_role": "identity_db",
                "table_key": "identity",
                "name": self.idx_identity_national,
            },
            {
                "key": "feature_vectors_method_created_at_desc",
                "database_role": "biometric_db",
                "table_key": "vectors",
                "name": self.idx_vector_method_created_at,
            },
            {
                "key": "method_retrieval_vectors_method_kind_created_at_desc",
                "database_role": "biometric_db",
                "table_key": "generic_vectors",
                "name": self.idx_generic_vector_method_kind_created_at,
            },
        ]
        for method in sorted(_legacy_storage_methods()):
            base.append(
                {
                    "key": f"feature_vectors_{method}_hnsw",
                    "database_role": "biometric_db",
                    "table_key": "vectors",
                    "name": self.idx_vector_hnsw[method],
                }
            )
        for dim in sorted(GENERIC_VECTOR_COLUMN_BY_DIM):
            base.append(
                {
                    "key": f"method_retrieval_vectors_{dim}_hnsw",
                    "database_role": "biometric_db",
                    "table_key": "generic_vectors",
                    "name": self.idx_generic_vector_hnsw[dim],
                }
            )
        return base

    def _identity_constraint_contract(self) -> List[Dict[str, str]]:
        return [
            {
                "key": "identity_full_name_not_null",
                "kind": "column_not_null",
                "column": "full_name",
            },
            {
                "key": "identity_name_norm_not_null",
                "kind": "column_not_null",
                "column": "name_norm",
            },
            {
                "key": "identity_full_name_not_blank_check",
                "kind": "check_constraint",
                "name": self.ck_identity_full_name_not_blank,
            },
            {
                "key": "identity_name_norm_not_blank_check",
                "kind": "check_constraint",
                "name": self.ck_identity_name_norm_not_blank,
            },
            {
                "key": "identity_name_norm_matches_full_name_check",
                "kind": "check_constraint",
                "name": self.ck_identity_name_norm_matches_full_name,
            },
            {
                "key": "identity_national_id_digits_only_check",
                "kind": "check_constraint",
                "name": self.ck_identity_national_id_digits_only,
            },
        ]

    def _collect_schema_index_presence(self, cur, *, table_presence: Dict[str, bool]) -> Dict[str, bool]:
        presence = {spec["key"]: False for spec in self._schema_index_contract()}
        for spec in self._schema_index_contract():
            if not table_presence.get(str(spec["table_key"]), False):
                continue
            presence[spec["key"]] = self._index_exists(cur, str(spec["name"]))
        return presence

    def _collect_identity_constraint_presence(self, cur) -> Dict[str, bool]:
        presence = {spec["key"]: False for spec in self._identity_constraint_contract()}
        for spec in self._identity_constraint_contract():
            if spec["kind"] == "column_not_null":
                presence[spec["key"]] = self._column_is_not_null(cur, self.identity_table, str(spec["column"]))
            else:
                presence[spec["key"]] = self._constraint_exists(cur, self.identity_table, str(spec["name"]))
        return presence

    def _missing_identity_contract_items(self, cur) -> List[str]:
        presence = self._collect_identity_constraint_presence(cur)
        return [key for key, present in presence.items() if not present]

    @staticmethod
    def _table_exists(cur, table_name: str) -> bool:
        cur.execute("SELECT to_regclass(%s) AS table_name", (table_name,))
        row = cur.fetchone()
        return bool(row and row["table_name"])

    @staticmethod
    def _index_exists(cur, index_name: str) -> bool:
        cur.execute("SELECT to_regclass(%s) AS table_name", (index_name,))
        row = cur.fetchone()
        return bool(row and row["table_name"])

    @staticmethod
    def _column_exists(cur, table_name: str, column_name: str) -> bool:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
            """,
            (table_name, column_name),
        )
        return cur.fetchone() is not None

    @staticmethod
    def _column_is_not_null(cur, table_name: str, column_name: str) -> bool:
        cur.execute(
            """
            SELECT is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
            """,
            (table_name, column_name),
        )
        row = cur.fetchone()
        return bool(row and str(row["is_nullable"]).upper() == "NO")

    @staticmethod
    def _constraint_exists(cur, table_name: str, constraint_name: str) -> bool:
        cur.execute(
            """
            SELECT 1
            FROM pg_catalog.pg_constraint AS c
            JOIN pg_catalog.pg_class AS t ON t.oid = c.conrelid
            JOIN pg_catalog.pg_namespace AS n ON n.oid = t.relnamespace
            WHERE n.nspname = current_schema()
              AND t.relname = %s
              AND c.conname = %s
            """,
            (table_name, constraint_name),
        )
        return cur.fetchone() is not None

    def _ensure_column_not_null(self, cur, *, table_name: str, column_name: str) -> None:
        if self._column_is_not_null(cur, table_name, column_name):
            return
        cur.execute(f"ALTER TABLE {table_name} ALTER COLUMN {column_name} SET NOT NULL")

    def _ensure_check_constraint(
        self,
        cur,
        *,
        table_name: str,
        constraint_name: str,
        check_sql: str,
    ) -> None:
        if self._constraint_exists(cur, table_name, constraint_name):
            return
        cur.execute(
            f"""
            ALTER TABLE {table_name}
            ADD CONSTRAINT {constraint_name}
            CHECK ({check_sql})
            """
        )

    def _read_identity_rows_from_table(self, cur, table_name: str) -> List[_IdentityMapRow]:
        if not self._table_exists(cur, table_name):
            return []
        has_full_name = self._column_exists(cur, table_name, 'full_name')
        has_name_norm = self._column_exists(cur, table_name, 'name_norm')
        select_full_name = 'full_name' if has_full_name else 'NULL AS full_name'
        select_name_norm = 'name_norm' if has_name_norm else 'NULL AS name_norm'
        cur.execute(
            f"""
            SELECT random_id,
                   national_id,
                   created_at,
                   {select_full_name},
                   {select_name_norm}
            FROM {table_name}
            WHERE national_id IS NOT NULL
              AND BTRIM(national_id) <> ''
              AND regexp_replace(COALESCE(national_id, ''), '\\D', '', 'g') <> ''
            ORDER BY random_id
            """
        )
        rows = cur.fetchall()
        return [self._row_to_identity_row(row) for row in rows]
    def _read_legacy_person_identity_rows(self, cur) -> List[_IdentityMapRow]:
        if not self._column_exists(cur, self.person_table, 'national_id'):
            return []
        has_full_name = self._column_exists(cur, self.person_table, 'full_name')
        has_name_norm = self._column_exists(cur, self.person_table, 'name_norm')
        select_full_name = 'full_name' if has_full_name else 'NULL AS full_name'
        select_name_norm = 'name_norm' if has_name_norm else 'NULL AS name_norm'
        cur.execute(
            f"""
            SELECT random_id,
                   regexp_replace(COALESCE(national_id, ''), '\\D', '', 'g') AS national_id,
                   created_at,
                   {select_full_name},
                   {select_name_norm}
            FROM {self.person_table}
            WHERE national_id IS NOT NULL
              AND BTRIM(national_id) <> ''
              AND regexp_replace(COALESCE(national_id, ''), '\\D', '', 'g') <> ''
            ORDER BY random_id
            """
        )
        rows = cur.fetchall()
        return [self._row_to_identity_row(row) for row in rows]
    def _lookup_random_id_with_cursor(self, cur, national_id_norm: str) -> Optional[str]:
        cur.execute(
            f"SELECT random_id FROM {self.identity_table} WHERE national_id = %s",
            (national_id_norm,),
        )
        row = cur.fetchone()
        return str(row["random_id"]) if row else None

    @classmethod
    def _ensure_unique_identity_rows(cls, rows: Sequence[_IdentityMapRow], *, context: str) -> None:
        seen_random_ids: Dict[str, str] = {}
        seen_national_ids: Dict[str, str] = {}
        dup_random: List[str] = []
        dup_national: List[str] = []
        for row in rows:
            if row.random_id in seen_random_ids and seen_random_ids[row.random_id] != row.national_id:
                dup_random.append(row.random_id)
            else:
                seen_random_ids[row.random_id] = row.national_id

            if row.national_id in seen_national_ids and seen_national_ids[row.national_id] != row.random_id:
                dup_national.append(row.national_id)
            else:
                seen_national_ids[row.national_id] = row.random_id

        if dup_random or dup_national:
            parts: List[str] = []
            if dup_random:
                parts.append("duplicate random_id values: " + ", ".join(sorted(set(dup_random))[:5]))
            if dup_national:
                parts.append("duplicate national_id values: " + ", ".join(sorted(set(dup_national))[:5]))
            raise RuntimeError(
                f"Automatic migration aborted because conflicting mappings were found in {context}: "
                + "; ".join(parts)
            )

    @staticmethod
    def _mapping_rows_equal(left: Sequence[_IdentityMapRow], right: Sequence[_IdentityMapRow]) -> bool:
        left_pairs = {(row.random_id, row.national_id) for row in left}
        right_pairs = {(row.random_id, row.national_id) for row in right}
        return left_pairs == right_pairs

    def _row_to_person_record(self, row) -> PersonDirectoryRecord:
        return PersonDirectoryRecord(
            random_id=str(row["random_id"]),
            full_name=str(row["full_name"]),
            name_norm=str(row["name_norm"]),
            national_id=str(row.get("national_id", "")),
            created_at=self._to_iso(row["created_at"]),
        )

    def _row_to_vector_record(self, row) -> FeatureVectorRecord:
        method = str(row["method"])
        spec = self._spec_for_method(method)
        payload = row.get(spec.generic_storage_column)
        if payload is None and spec.legacy_storage_column is not None:
            payload = row.get(spec.legacy_storage_column)
        vec = np.asarray(payload, dtype=np.float32).reshape(-1).copy()
        return FeatureVectorRecord(
            random_id=str(row["random_id"]),
            method=method,
            dim=int(row["dim"]),
            created_at=self._to_iso(row["created_at"]),
            vector=vec,
        )

    def _row_to_identity_row(self, row) -> _IdentityMapRow:
        national_id = normalize_national_id(row['national_id'])
        full_name = '' if row.get('full_name') is None else str(row.get('full_name')).strip()
        raw_name_norm = '' if row.get('name_norm') is None else str(row.get('name_norm')).strip()
        name_norm = raw_name_norm or (normalize_name(full_name) if full_name else '')
        return _IdentityMapRow(
            random_id=str(row['random_id']),
            national_id=national_id,
            created_at=self._as_utc_datetime(row['created_at']),
            full_name=full_name,
            name_norm=name_norm,
        )
    @staticmethod
    def _spec_for_method(method: str) -> VectorSpec:
        spec = VECTOR_SPECS.get(str(method).strip().lower())
        if spec is None:
            raise ValueError(unsupported_identification_retrieval_message(method))
        return spec
    def _prepare_vector(self, method: str, vec: np.ndarray) -> np.ndarray:
        spec = self._spec_for_method(method)
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size != spec.dim:
            raise ValueError(
                f"vector for method={method!r} has dim={arr.size}, expected exactly {spec.dim}"
            )
        calc = arr.astype(np.float64, copy=False)
        if not np.all(np.isfinite(calc)):
            return arr.astype(np.float32, copy=False)

        norm = float(np.linalg.norm(calc))
        if not np.isfinite(norm) or norm <= 1e-12:
            return arr.astype(np.float32, copy=False)

        return (calc / norm).astype(np.float32, copy=False)

    @staticmethod
    def _pattern_to_like(raw: str) -> str:
        s = str(raw).strip().replace("*", "%")
        if not s:
            return "%"
        if "%" not in s:
            s += "%"
        return s

    @staticmethod
    def _coerce_timestamp(raw: str | None) -> datetime:
        if not raw:
            return datetime.now(timezone.utc)
        value = str(raw).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @classmethod
    def _normalize_date_lower(cls, raw: str) -> datetime:
        value = str(raw).strip()
        if "T" in value:
            return cls._coerce_timestamp(value)
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = datetime.combine(dt.date(), time.min, tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @classmethod
    def _normalize_date_upper(cls, raw: str) -> datetime:
        value = str(raw).strip()
        if "T" in value:
            return cls._coerce_timestamp(value)
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = datetime.combine(dt.date(), time.max.replace(microsecond=0), tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _as_utc_datetime(value: object) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        raw = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _to_iso(value: object) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        return str(value)


    @staticmethod
    def _safe_rollback(conn) -> None:
        rollback = getattr(conn, 'rollback', None)
        if not callable(rollback):
            return
        with suppress(Exception):
            rollback()

    @staticmethod
    def _database_name_from_url(database_url: str) -> str:
        parsed = urlsplit(str(database_url))
        db_name = parsed.path.lstrip('/')
        return db_name or 'postgres'
    @staticmethod
    def _redacted_database_url(database_url: str) -> str:
        value = str(database_url)
        if "@" not in value or ":" not in value.split("@", 1)[0]:
            return value
        left, right = value.split("@", 1)
        if "://" in left:
            scheme, creds = left.split("://", 1)
            if ":" in creds:
                user, _ = creds.split(":", 1)
                return f"{scheme}://{user}:***@{right}"
        return value
