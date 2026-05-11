from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture()
def no_init(monkeypatch):
    monkeypatch.setattr(SecureSplitFingerprintStore, "_init_all", lambda self: None)


def test_dump_layout_reports_two_databases(no_init) -> None:
    store = SecureSplitFingerprintStore(
        database_url="postgresql://admin:secret@localhost:5432/biometric_db",
        identity_database_url="postgresql://admin:secret@localhost:5433/identity_db",
        table_prefix="demo",
    )

    layout = store.dump_layout()
    assert layout["dual_database_enabled"] == "true"
    assert layout["person_table"] == "biometric_db.demo_person_directory"
    assert layout["raw_fingerprints_table"] == "biometric_db.demo_raw_fingerprints"
    assert layout["feature_vectors_table"] == "biometric_db.demo_feature_vectors"
    assert layout["method_retrieval_vectors_table"] == "biometric_db.demo_method_retrieval_vectors"
    assert layout["identity_map_table"] == "identity_db.demo_identity_map"
    assert layout["method_generic_vectors_supported"] == "true"
    assert layout["biometric_database_url"].startswith("postgresql://admin:***@")
    assert layout["identity_database_url"].startswith("postgresql://admin:***@")


def test_duplicate_identity_rows_raise_clear_error() -> None:
    rows = [
        _IdentityMapRow("rid_a", "123456789", datetime.now(timezone.utc)),
        _IdentityMapRow("rid_b", "123456789", datetime.now(timezone.utc)),
    ]

    with pytest.raises(RuntimeError, match="duplicate national_id values"):
        SecureSplitFingerprintStore._ensure_unique_identity_rows(rows, context="legacy_person_table")


def test_mapping_row_comparison_ignores_created_at_differences() -> None:
    left = [
        _IdentityMapRow("rid_a", "111111111", datetime(2026, 3, 1, tzinfo=timezone.utc)),
        _IdentityMapRow("rid_b", "222222222", datetime(2026, 3, 2, tzinfo=timezone.utc)),
    ]
    right = [
        _IdentityMapRow("rid_b", "222222222", datetime(2026, 3, 10, tzinfo=timezone.utc)),
        _IdentityMapRow("rid_a", "111111111", datetime(2026, 3, 11, tzinfo=timezone.utc)),
    ]

    assert SecureSplitFingerprintStore._mapping_rows_equal(left, right) is True

from datetime import datetime, timezone

import numpy as np
import pytest

import src.fpbench.identification.secure_split_store as secure_store_module
from src.fpbench.identification.secure_split_store import (
    SecureSplitFingerprintStore,
    VectorSpec,
    _IdentityMapRow,
    _looks_like_missing_vector_type_error,
    resolve_biometric_database_url,
    resolve_identity_database_url,
)


def _generic_only_vector_spec(method: str = "future") -> VectorSpec:
    return VectorSpec(
        method=method,
        dim=512,
        vector_kind="future_embedding",
        distance_metric="cosine",
        generic_storage_column="vector_512",
        legacy_storage_column=None,
        legacy_storage_enabled=False,
        generic_storage_enabled=True,
        preferred_storage="generic",
    )


def test_compensation_is_invoked_if_biometric_commit_fails(monkeypatch):
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.person_table = "person_directory"
    store.identity_table = "identity_map"

    calls = {"compensate": 0}

    class DummyCursor:
        def __init__(self) -> None:
            self.last_call = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            self.last_call = (sql, params)

    class DummyConn:
        def __init__(self, fail_commit: bool = False):
            self.fail_commit = fail_commit
            self.commits = 0
            self.closed = False
            self.cursor_calls = 0

        def cursor(self):
            self.cursor_calls += 1
            return DummyCursor()

        def commit(self):
            self.commits += 1
            if self.fail_commit:
                raise RuntimeError("simulated biometric commit failure")

        def close(self):
            self.closed = True

    bio_conn = DummyConn(fail_commit=True)
    ident_conn = DummyConn(fail_commit=False)

    monkeypatch.setattr(store, "_connect_biometric", lambda: bio_conn)
    monkeypatch.setattr(store, "_connect_identity", lambda: ident_conn)
    monkeypatch.setattr(
        store,
        "_lookup_identity_row_by_national_id_with_cursor",
        lambda cur, nid: _IdentityMapRow("old_rid", nid, datetime.now(timezone.utc)),
    )
    monkeypatch.setattr(store, "_insert_person_row", lambda *a, **k: None)
    monkeypatch.setattr(store, "_insert_raw_row", lambda *a, **k: None)
    monkeypatch.setattr(store, "_insert_vector_rows", lambda *a, **k: None)
    monkeypatch.setattr(store, "_insert_identity_rows", lambda *a, **k: None)
    monkeypatch.setattr(store, "_safe_rollback", lambda conn: None)

    def _compensate(*, new_random_id: str, previous_mapping: _IdentityMapRow | None) -> None:
        _ = (new_random_id, previous_mapping)
        calls["compensate"] += 1

    monkeypatch.setattr(store, "_compensate_enroll_identity_failure", _compensate)

    with pytest.raises(RuntimeError) as excinfo:
        store._enroll_dual_database(
            full_name="Alice Levi",
            name_norm="alice levi",
            national_id_norm="123456789",
            capture_norm="plain",
            ext=".png",
            vector_payload={"dl": np.zeros(512, dtype=np.float32)},
            random_id="new_rid",
            created_at_dt=datetime.now(timezone.utc),
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            image_hash="abc",
            byte_size=3,
            replace_existing=True,
        )

    assert calls["compensate"] == 1
    message = str(excinfo.value)
    assert "Dual-database enroll failed." in message
    assert "Commit state: after identity_db committed; biometric_db had not committed successfully." in message
    assert "Manual reconciliation required: no." in message
    assert "scripts/diagnostics/reconcile_identification_runtime_db.py" in message
    assert "simulated biometric commit failure" in message


def test_insert_raw_row_writes_metadata_and_null_legacy_image_bytes() -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.raw_table = "raw_fingerprints"
    captured: dict[str, object] = {}

    class Cursor:
        def execute(self, sql, params=None):
            captured["sql"] = " ".join(str(sql).split())
            captured["params"] = params

    created_at = datetime.now(timezone.utc)

    store._insert_raw_row(
        Cursor(),
        random_id="rid_meta",
        capture="plain",
        ext=".png",
        sha256="abc",
        byte_size=3,
        created_at=created_at,
    )

    assert "image_bytes" in str(captured["sql"])
    assert "NULL" in str(captured["sql"])
    assert captured["params"] == ("rid_meta", "plain", ".png", "abc", 3, created_at)


def test_insert_vector_rows_dual_writes_legacy_and_generic_storage() -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.vector_table = "feature_vectors"
    store.generic_vector_table = "method_retrieval_vectors"
    captured: list[tuple[str, tuple[object, ...]]] = []

    class Cursor:
        def execute(self, sql, params=None):
            captured.append((" ".join(str(sql).split()), tuple(params or ())))

    created_at = datetime.now(timezone.utc)

    store._insert_vector_rows(
        Cursor(),
        random_id="rid_vector",
        created_at=created_at,
        vector_payload={
            "dl": np.ones(512, dtype=np.float32),
            "vit": np.ones(768, dtype=np.float32),
        },
    )

    assert len(captured) == 4
    generic_sql, generic_params = captured[0]
    legacy_sql, legacy_params = captured[1]
    assert "INSERT INTO method_retrieval_vectors" in generic_sql
    assert "vector_kind" in generic_sql
    assert generic_params[:6] == (
        "rid_vector",
        "dl",
        "deep_embedding_resnet",
        512,
        "cosine",
        created_at,
    )
    assert "INSERT INTO feature_vectors" in legacy_sql
    assert legacy_params[:4] == ("rid_vector", "dl", 512, created_at)
    assert captured[2][1][:6] == (
        "rid_vector",
        "vit",
        "vit_embedding",
        768,
        "cosine",
        created_at,
    )
    assert "INSERT INTO feature_vectors" in captured[3][0]


def test_insert_vector_rows_writes_classic_method_without_legacy_storage() -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.vector_table = "feature_vectors"
    store.generic_vector_table = "method_retrieval_vectors"
    captured: list[tuple[str, tuple[object, ...]]] = []

    class Cursor:
        def execute(self, sql, params=None):
            captured.append((" ".join(str(sql).split()), tuple(params or ())))

    created_at = datetime.now(timezone.utc)

    store._insert_vector_rows(
        Cursor(),
        random_id="rid_classic",
        created_at=created_at,
        vector_payload={"classic_orb": np.ones(512, dtype=np.float32)},
    )

    assert len(captured) == 1
    sql, params = captured[0]
    assert "INSERT INTO method_retrieval_vectors" in sql
    assert "INSERT INTO feature_vectors" not in sql
    assert params[:6] == (
        "rid_classic",
        "classic_orb",
        "classic_orb_aggregated_descriptor_v1",
        512,
        "cosine",
        created_at,
    )


def test_count_vectors_uses_generic_rows_for_classic_generic_only_method(monkeypatch) -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.generic_vector_table = "method_retrieval_vectors"
    store.vector_table = "feature_vectors"

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return Cursor()

    monkeypatch.setattr(store, "_connect_biometric", lambda: Conn())
    monkeypatch.setattr(store, "_table_exists", lambda cur, table_name: table_name == store.generic_vector_table)
    monkeypatch.setattr(store, "_generic_vector_schema_compatible", lambda cur: True)
    monkeypatch.setattr(
        store,
        "_count_generic_vector_rows_for_spec",
        lambda cur, *, spec, candidate_ids=None: 7,
    )
    monkeypatch.setattr(
        store,
        "_count_legacy_vector_rows_for_spec",
        lambda cur, *, spec, candidate_ids=None: (_ for _ in ()).throw(
            AssertionError("generic-only count should not inspect legacy rows")
        ),
    )

    assert store.count_vectors("classic_orb") == 7


def test_shortlist_reads_generic_storage_when_coverage_is_complete(monkeypatch) -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.generic_vector_table = "method_retrieval_vectors"
    executed: dict[str, object] = {}

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            executed["sql"] = " ".join(str(sql).split())
            executed["params"] = tuple(params or ())

        def fetchall(self):
            return [{"random_id": "rid_generic", "retrieval_score": 0.99}]

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return Cursor()

    monkeypatch.setattr(store, "_connect_biometric", lambda: Conn())
    monkeypatch.setattr(store, "_should_use_generic_vector_storage", lambda **kwargs: True)

    result = store.shortlist_by_vector(
        method="dl",
        probe_vector=np.ones(512, dtype=np.float32),
        limit=1,
    )

    assert result == [("rid_generic", pytest.approx(0.99))]
    assert "FROM method_retrieval_vectors" in str(executed["sql"])
    assert "vector_kind = %s" in str(executed["sql"])
    assert tuple(executed["params"])[1:5] == ("dl", "deep_embedding_resnet", 512, "cosine")


def test_shortlist_falls_back_to_legacy_when_generic_is_incomplete(monkeypatch) -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.generic_vector_table = "method_retrieval_vectors"
    store.vector_table = "feature_vectors"
    calls = {"legacy": 0}

    monkeypatch.setattr(store, "_should_use_generic_vector_storage", lambda **kwargs: False)

    def _legacy(**kwargs):
        _ = kwargs
        calls["legacy"] += 1
        return [("rid_legacy", 0.75)]

    monkeypatch.setattr(store, "_shortlist_by_legacy_vector", _legacy)

    result = store.shortlist_by_vector(
        method="dl",
        probe_vector=np.ones(512, dtype=np.float32),
        limit=1,
    )

    assert result == [("rid_legacy", 0.75)]
    assert calls["legacy"] == 1


def test_shortlist_classic_generic_only_method_never_falls_back_to_legacy(monkeypatch) -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.generic_vector_table = "method_retrieval_vectors"
    store.vector_table = "feature_vectors"
    executed: dict[str, object] = {}

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            executed["sql"] = " ".join(str(sql).split())
            executed["params"] = tuple(params or ())

        def fetchall(self):
            return [{"random_id": "rid_classic", "retrieval_score": 0.91}]

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return Cursor()

    monkeypatch.setattr(store, "_connect_biometric", lambda: Conn())
    monkeypatch.setattr(store, "_should_use_generic_vector_storage", lambda **kwargs: True)
    monkeypatch.setattr(
        store,
        "_shortlist_by_legacy_vector",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("generic-only shortlist should not use legacy storage")
        ),
    )

    result = store.shortlist_by_vector(
        method="classic_orb",
        probe_vector=np.ones(512, dtype=np.float32),
        limit=1,
    )

    assert result == [("rid_classic", pytest.approx(0.91))]
    assert "FROM method_retrieval_vectors" in str(executed["sql"])
    assert tuple(executed["params"])[1:5] == (
        "classic_orb",
        "classic_orb_aggregated_descriptor_v1",
        512,
        "cosine",
    )


def test_load_and_iter_vectors_support_classic_generic_only_method(monkeypatch) -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    created_at = datetime.now(timezone.utc)
    rows = [
        {
            "random_id": "rid_classic",
            "method": "classic_orb",
            "dim": 512,
            "created_at": created_at,
            "vector_512": [1.0] * 512,
            "vector_768": None,
        }
    ]

    monkeypatch.setattr(store, "_load_generic_vector_row", lambda **kwargs: rows[0])
    monkeypatch.setattr(store, "_load_generic_vector_rows", lambda **kwargs: rows)
    monkeypatch.setattr(
        store,
        "_connect_biometric",
        lambda: (_ for _ in ()).throw(
            AssertionError("generic-only vector reads should not open legacy fallback queries")
        ),
    )

    loaded = store.load_vector("rid_classic", "classic_orb")
    iterated = list(store.iter_vectors(["rid_classic"], "classic_orb"))

    assert loaded is not None
    assert loaded.random_id == "rid_classic"
    assert loaded.dim == 512
    assert len(iterated) == 1
    assert iterated[0].random_id == "rid_classic"


def test_backfill_generic_retrieval_vectors_is_idempotent(monkeypatch) -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.person_table = "person_directory"
    store.vector_table = "feature_vectors"
    store.generic_vector_table = "method_retrieval_vectors"
    store.table_prefix = ""
    store.idx_generic_vector_method_kind_created_at = "idx_method_retrieval_vectors_method_kind_created_at_desc"
    store.idx_generic_vector_hnsw = {
        512: "idx_method_retrieval_vectors_512_hnsw",
        768: "idx_method_retrieval_vectors_768_hnsw",
    }

    state = {
        "tables": {"person_directory", "feature_vectors", "method_retrieval_vectors"},
        "legacy": {
            "dl": {"rid_dl"},
            "vit": {"rid_vit"},
        },
        "generic": set(),
    }

    class Cursor:
        def __init__(self) -> None:
            self.rows: list[dict[str, object]] = []
            self.rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            query = " ".join(str(sql).split())
            params_tuple = tuple(params or ())
            self.rows = []
            self.rowcount = 0
            if query.startswith(("CREATE TABLE", "CREATE INDEX", "ALTER TABLE", "UPDATE", "SAVEPOINT", "RELEASE", "ROLLBACK")):
                return
            if "FROM pg_catalog.pg_extension" in query:
                self.rows = [{"has_vector": True}]
                return
            if "SELECT to_regclass(%s) AS table_name" in query:
                table = str(params_tuple[0])
                self.rows = [{"table_name": table if table in state["tables"] else None}]
                return
            if query.startswith("SELECT COUNT(*) AS n FROM feature_vectors"):
                method = str(params_tuple[0])
                self.rows = [{"n": len(state["legacy"][method])}]
                return
            if query.startswith("SELECT COUNT(*) AS n FROM method_retrieval_vectors"):
                method = str(params_tuple[0])
                vector_kind = str(params_tuple[1])
                self.rows = [
                    {
                        "n": len(
                            {
                                item
                                for item in state["generic"]
                                if item[1] == method and item[2] == vector_kind
                            }
                        )
                    }
                ]
                return
            if query.startswith("INSERT INTO method_retrieval_vectors"):
                vector_kind = str(params_tuple[0])
                method = str(params_tuple[3])
                before = len(state["generic"])
                for random_id in state["legacy"][method]:
                    state["generic"].add((random_id, method, vector_kind))
                self.rowcount = len(state["generic"]) - before
                return
            raise AssertionError(f"Unexpected SQL: {query}")

        def fetchone(self):
            return self.rows[0] if self.rows else None

    class Conn:
        def cursor(self):
            return Cursor()

        def commit(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(store, "_connect_biometric", lambda: Conn())
    monkeypatch.setattr(store, "_safe_rollback", lambda conn: None)

    first = store.backfill_generic_retrieval_vectors()
    second = store.backfill_generic_retrieval_vectors()

    assert first["copied_count"] == 2
    assert first["skipped_count"] == 0
    assert first["error_count"] == 0
    assert second["copied_count"] == 0
    assert second["skipped_count"] == 2
    assert second["status"] == "noop"


def test_environment_resolution_prefers_dual_database_test_urls(monkeypatch, no_init) -> None:
    monkeypatch.setenv(
        "IDENTIFICATION_TEST_BIOMETRIC_DATABASE_URL",
        "postgresql://admin:test@localhost:5432/biometric_db",
    )
    monkeypatch.setenv(
        "IDENTIFICATION_TEST_IDENTITY_DATABASE_URL",
        "postgresql://admin:test@localhost:5433/identity_db",
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("IDENTITY_DATABASE_URL", raising=False)

    store = SecureSplitFingerprintStore()

    assert store.biometric_database_url == "postgresql://admin:test@localhost:5432/biometric_db"
    assert store.identity_database_url == "postgresql://admin:test@localhost:5433/identity_db"
    assert store.dual_database_enabled is True


def test_environment_resolution_keeps_compatibility_when_identity_url_missing(monkeypatch, no_init) -> None:
    monkeypatch.setenv(
        "IDENTIFICATION_TEST_BIOMETRIC_DATABASE_URL",
        "postgresql://admin:test@localhost:5432/biometric_db",
    )
    monkeypatch.delenv("IDENTIFICATION_TEST_IDENTITY_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("IDENTITY_DATABASE_URL", raising=False)

    biometric_url = resolve_biometric_database_url()
    identity_url = resolve_identity_database_url(biometric_database_url=biometric_url)

    assert biometric_url == "postgresql://admin:test@localhost:5432/biometric_db"
    assert identity_url == biometric_url

    store = SecureSplitFingerprintStore()
    assert store.dual_database_enabled is False


def test_missing_vector_type_detection_matches_runtime_error_message() -> None:
    assert _looks_like_missing_vector_type_error(RuntimeError("vector type not found in the database")) is True
    assert _looks_like_missing_vector_type_error(RuntimeError('type "vector" does not exist')) is True
    assert _looks_like_missing_vector_type_error(RuntimeError("permission denied")) is False


def test_connect_postgres_bootstraps_pgvector_when_type_registration_fails(monkeypatch, no_init) -> None:
    store = SecureSplitFingerprintStore()

    executed_sql: list[str] = []
    committed = {"count": 0}
    closed = {"value": False}
    register_calls = {"count": 0}

    class DummyCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            _ = params
            executed_sql.append(sql)

    class DummyConn:
        def cursor(self):
            return DummyCursor()

        def commit(self):
            committed["count"] += 1

        def close(self):
            closed["value"] = True

    def _register_vector(conn):
        _ = conn
        register_calls["count"] += 1
        if register_calls["count"] == 1:
            raise RuntimeError("vector type not found in the database")

    monkeypatch.setattr(
        "src.fpbench.identification.secure_split_store._load_postgres_base_deps",
        lambda: (
            SimpleNamespace(connect=lambda *args, **kwargs: DummyConn()),
            None,
        ),
    )
    monkeypatch.setattr(
        "src.fpbench.identification.secure_split_store._load_pgvector_register",
        lambda: _register_vector,
    )

    conn = store._connect_postgres(
        "postgresql://admin:test@localhost:5432/biometric_db",
        autocommit=False,
        database_role="biometric",
        needs_vector=True,
    )

    assert isinstance(conn, DummyConn)
    assert register_calls["count"] == 2
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in sql for sql in executed_sql)
    assert committed["count"] == 1
    assert closed["value"] is False


def test_validate_dual_database_schema_state_uses_streaming_random_id_comparison(monkeypatch) -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.table_prefix = "demo_"
    store.person_table = "demo_person_directory"
    store.identity_table = "demo_identity_map"
    store.biometric_database_url = "postgresql://admin:test@localhost:5432/biometric_db"
    store.identity_database_url = "postgresql://admin:test@localhost:5433/identity_db"
    store.dual_database_enabled = True

    people_ids = [f"rid_{idx:04d}" for idx in range(6)]
    identity_ids = list(people_ids)
    fetchmany_calls = {"biometric_db": 0, "identity_db": 0}

    class StreamingCursor:
        def __init__(self, *, role: str) -> None:
            self.role = role
            self.rows: list[dict[str, object]] = []
            self.offset = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            _ = params
            query = " ".join(str(sql).split())
            self.offset = 0
            if query == f"SELECT COUNT(*) AS n FROM {store.person_table}":
                self.rows = [{"n": len(people_ids)}]
                return
            if query == f"SELECT COUNT(*) AS n FROM {store.identity_table}":
                self.rows = [{"n": len(identity_ids)}]
                return
            if query == f"SELECT random_id FROM {store.person_table} ORDER BY random_id":
                self.rows = [{"random_id": rid} for rid in people_ids]
                return
            if query == f"SELECT random_id FROM {store.identity_table} ORDER BY random_id":
                self.rows = [{"random_id": rid} for rid in identity_ids]
                return
            raise AssertionError(f"Unexpected SQL: {query}")

        def fetchone(self):
            if self.offset >= len(self.rows):
                return None
            row = self.rows[self.offset]
            self.offset += 1
            return row

        def fetchmany(self, size):
            fetchmany_calls[self.role] += 1
            rows = list(self.rows[self.offset : self.offset + int(size)])
            self.offset += len(rows)
            return rows

        def fetchall(self):
            raise AssertionError("streaming validation should not fetch all random_id rows at once")

    class StreamingConn:
        def __init__(self, *, role: str) -> None:
            self.role = role

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self, *args, **kwargs):
            _ = (args, kwargs)
            return StreamingCursor(role=self.role)

    monkeypatch.setattr(store, "_connect_biometric", lambda: StreamingConn(role="biometric_db"))
    monkeypatch.setattr(store, "_connect_identity", lambda: StreamingConn(role="identity_db"))
    monkeypatch.setattr(
        store,
        "_connect_postgres_inspection",
        lambda database_url, *, database_role: StreamingConn(role=database_role),
    )
    monkeypatch.setattr(store, "_column_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(store, "_table_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(store, "_legacy_person_profile_columns_exist", lambda *args, **kwargs: False)
    monkeypatch.setattr(store, "_identity_rows_missing_profile", lambda *args, **kwargs: [])
    monkeypatch.setattr(store, "_missing_identity_contract_items", lambda *args, **kwargs: [])

    store._validate_dual_database_schema_state()

    assert fetchmany_calls["biometric_db"] > 0
    assert fetchmany_calls["identity_db"] > 0


def test_validate_dual_database_schema_state_wraps_failures_with_reconciliation_guidance(monkeypatch) -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.table_prefix = "demo_"
    store.person_table = "demo_person_directory"
    store.identity_table = "demo_identity_map"
    store.biometric_database_url = "postgresql://admin:test@localhost:5432/biometric_db"
    store.identity_database_url = "postgresql://admin:test@localhost:5433/identity_db"
    store.dual_database_enabled = True

    class CountCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            _ = params
            query = " ".join(str(sql).split())
            if store.person_table in query:
                self.row = {"n": 2}
            elif store.identity_table in query:
                self.row = {"n": 1}
            else:
                raise AssertionError(f"Unexpected SQL: {query}")

        def fetchone(self):
            return self.row

    class CountConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return CountCursor()

    monkeypatch.setattr(store, "_connect_biometric", lambda: CountConn())
    monkeypatch.setattr(store, "_connect_identity", lambda: CountConn())
    monkeypatch.setattr(store, "_column_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(store, "_table_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(store, "_legacy_person_profile_columns_exist", lambda *args, **kwargs: False)
    monkeypatch.setattr(store, "_identity_rows_missing_profile", lambda *args, **kwargs: [])
    monkeypatch.setattr(store, "_missing_identity_contract_items", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        store,
        "_inspect_dual_database_identity_drift",
        lambda **kwargs: SimpleNamespace(
            people_without_identity_sample=["rid_missing"],
            identity_without_people_sample=[],
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        store._validate_dual_database_schema_state()

    message = str(excinfo.value)
    assert "Dual-database startup_validation failed." in message
    assert "Commit state: validation/read-only check; no side committed during the failing step." in message
    assert "Manual reconciliation required: yes." in message
    assert "scripts/diagnostics/reconcile_identification_runtime_db.py" in message
    assert "rid_missing" in message


def test_purge_dual_database_failure_message_includes_commit_state_and_command(monkeypatch) -> None:
    store = SecureSplitFingerprintStore.__new__(SecureSplitFingerprintStore)
    store.table_prefix = "demo_"
    store.person_table = "demo_person_directory"
    store.identity_table = "demo_identity_map"
    store.dual_database_enabled = True

    restore_calls = {"count": 0}

    class DummyCursor:
        def __init__(self) -> None:
            self.rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            _ = (sql, params)
            self.rowcount = 1

    class DummyConn:
        def __init__(self, fail_commit: bool = False):
            self.fail_commit = fail_commit
            self.closed = False

        def cursor(self):
            return DummyCursor()

        def commit(self):
            if self.fail_commit:
                raise RuntimeError("simulated biometric commit failure")

        def close(self):
            self.closed = True

    bio_conn = DummyConn(fail_commit=True)
    ident_conn = DummyConn(fail_commit=False)

    monkeypatch.setattr(store, "_connect_biometric", lambda: bio_conn)
    monkeypatch.setattr(store, "_connect_identity", lambda: ident_conn)
    monkeypatch.setattr(
        store,
        "_lookup_identity_row_by_random_id_with_cursor",
        lambda cur, random_id: _IdentityMapRow("rid_existing", "123456789", datetime.now(timezone.utc)),
    )
    monkeypatch.setattr(store, "_safe_rollback", lambda conn: None)

    def _restore(row: _IdentityMapRow) -> None:
        _ = row
        restore_calls["count"] += 1

    monkeypatch.setattr(store, "_restore_identity_row", _restore)

    with pytest.raises(RuntimeError) as excinfo:
        store._purge_dual_database("rid_existing")

    assert restore_calls["count"] == 1
    message = str(excinfo.value)
    assert "Dual-database purge failed." in message
    assert "Commit state: after identity_db committed; biometric_db had not committed successfully." in message
    assert "Manual reconciliation required: no." in message
    assert "scripts/diagnostics/reconcile_identification_runtime_db.py" in message
    assert "simulated biometric commit failure" in message
