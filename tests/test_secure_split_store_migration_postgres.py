from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg", reason="PostgreSQL integration test requires psycopg")
pytest.importorskip("pgvector", reason="PostgreSQL integration test requires pgvector")

RUNBOOK_PATH = Path(__file__).resolve().parents[1] / "LOCAL_DUAL_DB_RUNBOOK.md"
LOCAL_SETUP_HINT = (
    "Dual-database migration tests require two distinct PostgreSQL URLs.\n"
    "PowerShell quick start:\n"
    "  docker compose -f apps/api/docker-compose.yml up -d biometric_db identity_db\n"
    '  $env:IDENTIFICATION_TEST_BIOMETRIC_DATABASE_URL = "postgresql://admin:biometric_secret@127.0.0.1:5432/biometric_db"\n'
    '  $env:IDENTIFICATION_TEST_IDENTITY_DATABASE_URL = "postgresql://admin:identity_secret@127.0.0.1:5433/identity_db"\n'
    f"See {RUNBOOK_PATH.name} for the full local runbook."
)

BIOMETRIC_DB_URL = (
    os.environ.get("IDENTIFICATION_TEST_BIOMETRIC_DATABASE_URL")
    or os.environ.get("IDENTIFICATION_TEST_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
)
IDENTITY_DB_URL = (
    os.environ.get("IDENTIFICATION_TEST_IDENTITY_DATABASE_URL")
    or os.environ.get("IDENTITY_DATABASE_URL")
)

if not BIOMETRIC_DB_URL or not IDENTITY_DB_URL:
    pytest.skip(LOCAL_SETUP_HINT, allow_module_level=True)

if BIOMETRIC_DB_URL == IDENTITY_DB_URL:
    pytest.skip(
        LOCAL_SETUP_HINT
        + "\nThe configured biometric and identity URLs are identical; set them to different PostgreSQL databases.",
        allow_module_level=True,
    )

from src.fpbench.identification.secure_split_store import SecureSplitFingerprintStore


def _exec(database_url: str, sql: str, params=None) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())


def _fetch_one(database_url: str, sql: str, params=None):
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()


def _relation_exists(database_url: str, relation_name: str) -> bool:
    row = _fetch_one(database_url, "SELECT to_regclass(%s)", (relation_name,))
    return bool(row and row[0] is not None)


def _column_is_not_null(database_url: str, table_name: str, column_name: str) -> bool:
    row = _fetch_one(
        database_url,
        """
        SELECT is_nullable
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = %s
        """,
        (table_name, column_name),
    )
    return bool(row and str(row[0]).upper() == "NO")


def _constraint_exists(database_url: str, table_name: str, constraint_name: str) -> bool:
    row = _fetch_one(
        database_url,
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
    return bool(row)


def test_fresh_schema_reports_hardening_contract_and_supporting_indexes() -> None:
    prefix = f"hard_{uuid.uuid4().hex[:8]}"

    store = SecureSplitFingerprintStore(
        BIOMETRIC_DB_URL,
        identity_database_url=IDENTITY_DB_URL,
        table_prefix=prefix,
    )

    assert _relation_exists(BIOMETRIC_DB_URL, store.idx_person_created_at)
    assert _relation_exists(BIOMETRIC_DB_URL, store.idx_vector_method_created_at)
    assert _relation_exists(BIOMETRIC_DB_URL, store.idx_vector_dl)
    assert _relation_exists(BIOMETRIC_DB_URL, store.idx_vector_vit)
    assert _relation_exists(IDENTITY_DB_URL, store.idx_identity_name)
    assert _relation_exists(IDENTITY_DB_URL, store.idx_identity_national)

    assert _column_is_not_null(IDENTITY_DB_URL, store.identity_table, "full_name") is True
    assert _column_is_not_null(IDENTITY_DB_URL, store.identity_table, "name_norm") is True
    assert _constraint_exists(IDENTITY_DB_URL, store.identity_table, store.ck_identity_full_name_not_blank) is True
    assert _constraint_exists(IDENTITY_DB_URL, store.identity_table, store.ck_identity_name_norm_not_blank) is True
    assert (
        _constraint_exists(
            IDENTITY_DB_URL,
            store.identity_table,
            store.ck_identity_name_norm_matches_full_name,
        )
        is True
    )
    assert _constraint_exists(IDENTITY_DB_URL, store.identity_table, store.ck_identity_national_id_digits_only) is True

    payload = SecureSplitFingerprintStore.inspect_runtime_state(
        database_url=BIOMETRIC_DB_URL,
        identity_database_url=IDENTITY_DB_URL,
        table_prefix=prefix,
    )

    assert payload["schema_hardening"]["identity_map_guarantees"]["contract_enforced"] is True
    assert payload["schema_hardening"]["identity_map_guarantees"]["completeness_guaranteed"] is True
    assert payload["schema_hardening"]["drift"]["missing_indexes"] == []
    assert payload["schema_hardening"]["drift"]["missing_constraints"] == []


def test_single_database_hardening_normalizes_existing_identity_rows() -> None:
    prefix = f"hardmig_{uuid.uuid4().hex[:8]}"
    person_table = f"{prefix}_person_directory"
    identity_table = f"{prefix}_identity_map"

    _exec(
        BIOMETRIC_DB_URL,
        f"""
        CREATE TABLE {person_table} (
            random_id TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
    )
    _exec(
        BIOMETRIC_DB_URL,
        f"""
        CREATE TABLE {identity_table} (
            random_id TEXT PRIMARY KEY REFERENCES {person_table}(random_id) ON DELETE CASCADE,
            national_id TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            full_name TEXT,
            name_norm TEXT
        )
        """,
    )
    _exec(
        BIOMETRIC_DB_URL,
        f"""
        INSERT INTO {person_table} (random_id, created_at)
        VALUES ('rid_norm', '2026-03-18T10:00:00+00:00')
        """,
    )
    _exec(
        BIOMETRIC_DB_URL,
        f"""
        INSERT INTO {identity_table} (random_id, national_id, created_at, full_name, name_norm)
        VALUES ('rid_norm', '123-456-789', '2026-03-18T10:00:00+00:00', '  Alice   Levi  ', 'WRONG')
        """,
    )

    store = SecureSplitFingerprintStore(
        BIOMETRIC_DB_URL,
        identity_database_url=BIOMETRIC_DB_URL,
        table_prefix=prefix,
    )

    person = store.get_person("rid_norm")
    assert person is not None
    assert person.full_name == "Alice   Levi"
    assert person.name_norm == "alice levi"
    assert person.national_id == "123456789"
    assert _column_is_not_null(BIOMETRIC_DB_URL, identity_table, "full_name") is True
    assert _column_is_not_null(BIOMETRIC_DB_URL, identity_table, "name_norm") is True


def test_single_database_hardening_fails_on_invalid_legacy_national_id() -> None:
    prefix = f"hardfail_{uuid.uuid4().hex[:8]}"
    person_table = f"{prefix}_person_directory"
    identity_table = f"{prefix}_identity_map"

    _exec(
        BIOMETRIC_DB_URL,
        f"""
        CREATE TABLE {person_table} (
            random_id TEXT PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
    )
    _exec(
        BIOMETRIC_DB_URL,
        f"""
        CREATE TABLE {identity_table} (
            random_id TEXT PRIMARY KEY REFERENCES {person_table}(random_id) ON DELETE CASCADE,
            national_id TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            full_name TEXT,
            name_norm TEXT
        )
        """,
    )
    _exec(
        BIOMETRIC_DB_URL,
        f"""
        INSERT INTO {person_table} (random_id, created_at)
        VALUES ('rid_bad', '2026-03-18T10:00:00+00:00')
        """,
    )
    _exec(
        BIOMETRIC_DB_URL,
        f"""
        INSERT INTO {identity_table} (random_id, national_id, created_at, full_name, name_norm)
        VALUES ('rid_bad', 'ABC', '2026-03-18T10:00:00+00:00', 'Alice Levi', 'alice levi')
        """,
    )

    with pytest.raises(RuntimeError, match="blank after normalization"):
        SecureSplitFingerprintStore(
            BIOMETRIC_DB_URL,
            identity_database_url=BIOMETRIC_DB_URL,
            table_prefix=prefix,
        )


def test_migration_from_legacy_same_db_identity_table() -> None:
    prefix = f"mig_{uuid.uuid4().hex[:8]}"
    person_table = f"{prefix}_person_directory"
    identity_table = f"{prefix}_identity_map"

    _exec(
        BIOMETRIC_DB_URL,
        f"""
        CREATE TABLE {person_table} (
            random_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            name_norm TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
    )
    _exec(
        BIOMETRIC_DB_URL,
        f"""
        CREATE TABLE {identity_table} (
            random_id TEXT PRIMARY KEY REFERENCES {person_table}(random_id) ON DELETE CASCADE,
            national_id TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
    )
    _exec(
        BIOMETRIC_DB_URL,
        f"""
        INSERT INTO {person_table} (random_id, full_name, name_norm, created_at)
        VALUES
            ('rid_a', 'Alice Levi', 'alice levi', '2026-03-18T10:00:00+00:00'),
            ('rid_b', 'Bob Cohen', 'bob cohen', '2026-03-18T11:00:00+00:00')
        """,
    )
    _exec(
        BIOMETRIC_DB_URL,
        f"""
        INSERT INTO {identity_table} (random_id, national_id, created_at)
        VALUES
            ('rid_a', '111111111', '2026-03-18T10:00:00+00:00'),
            ('rid_b', '222222222', '2026-03-18T11:00:00+00:00')
        """,
    )

    store = SecureSplitFingerprintStore(
        BIOMETRIC_DB_URL,
        identity_database_url=IDENTITY_DB_URL,
        table_prefix=prefix,
    )

    person = store.get_person("rid_a")
    assert person is not None
    assert person.national_id == "111111111"
    assert store.dump_layout()["dual_database_enabled"] == "true"

    legacy_exists = _fetch_one(
        BIOMETRIC_DB_URL,
        "SELECT to_regclass(%s)",
        (identity_table,),
    )
    assert legacy_exists[0] is None

    identity_count = _fetch_one(
        IDENTITY_DB_URL,
        f"SELECT COUNT(*) FROM {identity_table}",
    )
    assert int(identity_count[0]) == 2
    person_full_name_exists = _fetch_one(
        BIOMETRIC_DB_URL,
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = 'full_name'
        """,
        (person_table,),
    )
    assert person_full_name_exists is None

    identity_profile = _fetch_one(
        IDENTITY_DB_URL,
        f"SELECT full_name, name_norm FROM {identity_table} WHERE random_id = 'rid_a'",
    )
    assert identity_profile == ("Alice Levi", "alice levi")


def test_migration_from_legacy_person_directory_column() -> None:
    prefix = f"migcol_{uuid.uuid4().hex[:8]}"
    person_table = f"{prefix}_person_directory"

    _exec(
        BIOMETRIC_DB_URL,
        f"""
        CREATE TABLE {person_table} (
            random_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            name_norm TEXT NOT NULL,
            national_id TEXT,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
    )
    _exec(
        BIOMETRIC_DB_URL,
        f"""
        INSERT INTO {person_table} (random_id, full_name, name_norm, national_id, created_at)
        VALUES ('rid_c', 'Carol Ben', 'carol ben', '333-333-333', '2026-03-18T12:00:00+00:00')
        """,
    )

    store = SecureSplitFingerprintStore(
        BIOMETRIC_DB_URL,
        identity_database_url=IDENTITY_DB_URL,
        table_prefix=prefix,
    )

    person = store.get_person("rid_c")
    assert person is not None
    assert person.national_id == "333333333"
    assert store.dump_layout()["dual_database_enabled"] == "true"

    column_exists = _fetch_one(
        BIOMETRIC_DB_URL,
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = 'national_id'
        """,
        (person_table,),
    )
    assert column_exists is None
    person_full_name_exists = _fetch_one(
        BIOMETRIC_DB_URL,
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = 'full_name'
        """,
        (person_table,),
    )
    assert person_full_name_exists is None
    identity_profile = _fetch_one(
        IDENTITY_DB_URL,
        f"SELECT full_name, name_norm FROM {prefix}_identity_map WHERE random_id = 'rid_c'",
    )
    assert identity_profile == ("Carol Ben", "carol ben")
