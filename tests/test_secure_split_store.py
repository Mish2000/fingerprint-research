from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg", reason="PostgreSQL integration test requires psycopg")
pytest.importorskip("pgvector", reason="PostgreSQL integration test requires pgvector")

RUNBOOK_PATH = Path(__file__).resolve().parents[1] / "LOCAL_DUAL_DB_RUNBOOK.md"
LOCAL_SETUP_HINT = (
    "Set PostgreSQL URLs for the biometric and identity databases before running this test.\n"
    "PowerShell quick start:\n"
    '  $env:BIOMETRIC_POSTGRES_PASSWORD = "change_me_biometric_dev_password"\n'
    '  $env:IDENTITY_POSTGRES_PASSWORD = "change_me_identity_dev_password"\n'
    "  docker compose -f apps/api/docker-compose.yml up -d biometric_db identity_db\n"
    '  $env:IDENTIFICATION_TEST_BIOMETRIC_DATABASE_URL = "postgresql://admin:$env:BIOMETRIC_POSTGRES_PASSWORD@127.0.0.1:5432/biometric_db"\n'
    '  $env:IDENTIFICATION_TEST_IDENTITY_DATABASE_URL = "postgresql://admin:$env:IDENTITY_POSTGRES_PASSWORD@127.0.0.1:5433/identity_db"\n'
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
    or BIOMETRIC_DB_URL
)
if not BIOMETRIC_DB_URL:
    pytest.skip(LOCAL_SETUP_HINT, allow_module_level=True)

from src.fpbench.identification.secure_split_store import IdentifyHints, SecureSplitFingerprintStore


def _fetch_one(database_url: str, sql: str, params=None):
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()


def test_secure_split_store_enroll_search_and_purge() -> None:
    store = SecureSplitFingerprintStore(
        BIOMETRIC_DB_URL,
        identity_database_url=IDENTITY_DB_URL,
        table_prefix=f"test_{uuid.uuid4().hex[:8]}",
    )

    r1 = store.enroll(
        full_name="Michael Sirak",
        national_id="123456789",
        image_bytes=b"fingerprint-a",
        capture="plain",
        ext=".png",
        vectors={"dl": np.array([1.0] * 512, dtype=np.float32)},
    )
    r2 = store.enroll(
        full_name="Mina Cohen",
        national_id="987654321",
        image_bytes=b"fingerprint-b",
        capture="roll",
        ext=".png",
        vectors={
            "dl": np.array([0.0] * 511 + [1.0], dtype=np.float32),
            "vit": np.array([0.5] * 768, dtype=np.float32),
        },
    )

    layout = store.dump_layout()
    assert layout["biometric_database_url"]
    assert layout["identity_database_url"]
    assert layout["dual_database_enabled"] == ("true" if BIOMETRIC_DB_URL != IDENTITY_DB_URL else "false")
    assert layout["person_table"].endswith(store.person_table)
    assert layout["identity_map_table"].endswith(store.identity_table)

    assert store.total_people() == 2
    assert store.count_vectors("dl") == 2
    generic_counts = _fetch_one(
        BIOMETRIC_DB_URL,
        f"""
        SELECT
            COUNT(*) FILTER (WHERE method = 'dl' AND vector_kind = 'deep_embedding_resnet') AS dl_count,
            COUNT(*) FILTER (WHERE method = 'vit' AND vector_kind = 'vit_embedding') AS vit_count
        FROM {store.generic_vector_table}
        """,
    )
    assert generic_counts == (2, 1)

    name_hits = store.search_people(IdentifyHints(name_pattern="mi*"))
    assert {item.full_name for item in name_hits} == {"Michael Sirak", "Mina Cohen"}

    id_hits = store.search_people(IdentifyHints(national_id_pattern="123*"))
    assert len(id_hits) == 1
    assert id_hits[0].random_id == r1.random_id

    person = store.get_person(r2.random_id)
    bio_full_name_col = _fetch_one(
        BIOMETRIC_DB_URL,
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = 'full_name'
        """,
        (store.person_table,),
    )
    assert bio_full_name_col is None

    identity_name_col = _fetch_one(
        IDENTITY_DB_URL,
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = 'full_name'
        """,
        (store.identity_table,),
    )
    assert identity_name_col is not None
    assert person is not None
    assert person.national_id == "987654321"

    raw = store.load_raw_fingerprint(r2.random_id)
    assert raw is not None
    assert raw.capture == "roll"
    assert raw.sha256 == hashlib.sha256(b"fingerprint-b").hexdigest()
    assert raw.byte_size == len(b"fingerprint-b")
    assert raw.legacy_image_bytes_present is False
    persisted_raw = _fetch_one(
        BIOMETRIC_DB_URL,
        f"SELECT image_bytes, byte_size FROM {store.raw_table} WHERE random_id = %s",
        (r2.random_id,),
    )
    assert persisted_raw is not None
    assert persisted_raw[0] is None
    assert int(persisted_raw[1]) == len(b"fingerprint-b")

    vec = store.load_vector(r2.random_id, "vit")
    assert vec is not None
    assert vec.dim == 768
    expected = np.array([0.5] * 768, dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(vec.vector, expected)

    shortlist = store.shortlist_by_vector(
        method="dl",
        probe_vector=np.array([1.0] * 512, dtype=np.float32),
        limit=2,
    )
    assert shortlist[0][0] == r1.random_id

    with psycopg.connect(BIOMETRIC_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {store.generic_vector_table} WHERE method = 'dl' AND vector_kind = 'deep_embedding_resnet'"
            )
    fallback_shortlist = store.shortlist_by_vector(
        method="dl",
        probe_vector=np.array([1.0] * 512, dtype=np.float32),
        limit=2,
    )
    assert fallback_shortlist[0][0] == r1.random_id

    backfill = store.backfill_generic_retrieval_vectors()
    assert backfill["copied_count"] == 2
    assert backfill["skipped_count"] == 1
    second_backfill = store.backfill_generic_retrieval_vectors()
    assert second_backfill["copied_count"] == 0
    assert second_backfill["skipped_count"] == 3

    removed = store.purge(r1.random_id)
    assert removed is True
    assert store.get_person(r1.random_id) is None
    assert store.load_raw_fingerprint(r1.random_id) is None
    assert store.load_vector(r1.random_id, "dl") is None
