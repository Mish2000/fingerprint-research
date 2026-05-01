from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from src.fpbench.identification.secure_split_store import SecureSplitFingerprintStore

_COUNT_SQL_RE = re.compile(r"SELECT COUNT\(\*\) AS n FROM ([A-Za-z0-9_]+)")
_IDS_SQL_RE = re.compile(r"SELECT random_id FROM ([A-Za-z0-9_]+) ORDER BY random_id")
_ANTI_JOIN_COUNT_RE = re.compile(
    r"SELECT COUNT\(\*\) AS n FROM ([A-Za-z0-9_]+) AS ([A-Za-z]) WHERE NOT EXISTS "
    r"\( SELECT 1 FROM ([A-Za-z0-9_]+) AS ([A-Za-z]) WHERE \4\.random_id = \2\.random_id \)"
)
_ANTI_JOIN_SAMPLE_RE = re.compile(
    r"SELECT ([A-Za-z])\.random_id FROM ([A-Za-z0-9_]+) AS ([A-Za-z]) WHERE NOT EXISTS "
    r"\( SELECT 1 FROM ([A-Za-z0-9_]+) AS ([A-Za-z]) WHERE \5\.random_id = \3\.random_id \) "
    r"ORDER BY \1\.random_id LIMIT %s"
)
_DELETE_ANTI_JOIN_RE = re.compile(
    r"DELETE FROM ([A-Za-z0-9_]+) AS ([A-Za-z]) WHERE NOT EXISTS "
    r"\( SELECT 1 FROM ([A-Za-z0-9_]+) AS ([A-Za-z]) WHERE \4\.random_id = \2\.random_id \)"
)
_SELECT_ANY_RE = re.compile(
    r"SELECT random_id FROM ([A-Za-z0-9_]+) WHERE random_id = ANY\(%s\) ORDER BY random_id"
)
_DELETE_ANY_RE = re.compile(r"DELETE FROM ([A-Za-z0-9_]+) WHERE random_id = ANY\(%s\)")


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{script_name.replace('.', '_')}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inspection_state(
    prefix: str,
    *,
    identity_table_present: bool = True,
    include_biometric_tables: bool = True,
    person_ids: list[str] | None = None,
    identity_ids: list[str] | None = None,
    raw_orphan_ids: list[str] | None = None,
    vector_orphan_ids: list[str] | None = None,
) -> dict[str, object]:
    person_table = f"{prefix}person_directory"
    identity_table = f"{prefix}identity_map"
    raw_table = f"{prefix}raw_fingerprints"
    vector_table = f"{prefix}feature_vectors"
    person_id_values = list(person_ids) if person_ids is not None else (
        ["rid_a", "rid_b"] if include_biometric_tables else []
    )
    identity_id_values = list(identity_ids) if identity_ids is not None else (
        ["rid_a", "rid_b"] if identity_table_present else []
    )
    raw_orphan_values = list(raw_orphan_ids or [])
    vector_orphan_values = list(vector_orphan_ids or [])
    return {
        "tables": {
            person_table: include_biometric_tables,
            identity_table: identity_table_present,
            raw_table: include_biometric_tables,
            vector_table: include_biometric_tables,
        },
        "columns": {
            (person_table, "national_id"): False,
            (person_table, "full_name"): False,
            (person_table, "name_norm"): False,
        },
        "column_nullability": {
            (identity_table, "full_name"): identity_table_present,
            (identity_table, "name_norm"): identity_table_present,
            (identity_table, "national_id"): identity_table_present,
        },
        "indexes": {
            f"{prefix}idx_person_created_at_desc": include_biometric_tables,
            f"{prefix}idx_identity_name_norm_prefix": identity_table_present,
            f"{prefix}idx_identity_national_id_prefix": identity_table_present,
            f"{prefix}idx_feature_vectors_method_created_at_desc": include_biometric_tables,
            f"{prefix}idx_feature_vectors_dl_hnsw": include_biometric_tables,
            f"{prefix}idx_feature_vectors_vit_hnsw": include_biometric_tables,
        },
        "constraints": {
            (identity_table, f"{prefix}ck_identity_full_name_not_blank"): identity_table_present,
            (identity_table, f"{prefix}ck_identity_name_norm_not_blank"): identity_table_present,
            (identity_table, f"{prefix}ck_identity_name_norm_matches_full_name"): identity_table_present,
            (identity_table, f"{prefix}ck_identity_national_id_digits_only"): identity_table_present,
        },
        "counts": {
            person_table: len(person_id_values) if include_biometric_tables else None,
            identity_table: len(identity_id_values) if identity_table_present else None,
            raw_table: (len(person_id_values) + len(raw_orphan_values)) if include_biometric_tables else None,
        },
        "ids": {
            person_table: person_id_values if include_biometric_tables else [],
            identity_table: identity_id_values if identity_table_present else [],
        },
        "vector_counts": {"dl": 2, "vit": 1} if include_biometric_tables else {},
        "vector_extension": True,
        "identity_missing_profile": [],
        "orphans": {
            raw_table: raw_orphan_values if include_biometric_tables else [],
            vector_table: vector_orphan_values if include_biometric_tables else [],
        },
    }


class FakeInspectionCursor:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state
        self._rows: list[dict[str, object]] = []
        self._offset = 0
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        query = " ".join(str(sql).split())
        params = tuple(params or ())
        self._offset = 0
        self.rowcount = 0

        if "FROM pg_catalog.pg_extension" in query:
            self._rows = [{"has_vector": bool(self.state["vector_extension"])}]
            return

        if "SELECT to_regclass(%s) AS table_name" in query:
            table_name = str(params[0])
            exists = bool(self.state["tables"].get(table_name, False) or self.state["indexes"].get(table_name, False))
            self._rows = [{"table_name": table_name if exists else None}]
            return

        if "SELECT is_nullable" in query and "FROM information_schema.columns" in query:
            table_name, column_name = str(params[0]), str(params[1])
            if (table_name, column_name) not in self.state["column_nullability"]:
                self._rows = []
            else:
                is_not_null = bool(self.state["column_nullability"][(table_name, column_name)])
                self._rows = [{"is_nullable": "NO" if is_not_null else "YES"}]
            return

        if "FROM information_schema.columns" in query:
            table_name, column_name = str(params[0]), str(params[1])
            exists = bool(self.state["columns"].get((table_name, column_name), False))
            self._rows = [{"column_name": column_name}] if exists else []
            return

        if "FROM pg_catalog.pg_constraint AS c" in query:
            table_name, constraint_name = str(params[0]), str(params[1])
            exists = bool(self.state["constraints"].get((table_name, constraint_name), False))
            self._rows = [{"exists": 1}] if exists else []
            return

        if "GROUP BY method" in query:
            self._rows = [
                {"method": method, "n": count}
                for method, count in sorted(dict(self.state["vector_counts"]).items())
            ]
            return

        if "WHERE full_name IS NULL OR BTRIM(full_name) = ''" in query:
            if query.startswith("SELECT COUNT(*) AS n"):
                self._rows = [{"n": len(list(self.state["identity_missing_profile"]))}]
            else:
                self._rows = [{"random_id": rid} for rid in list(self.state["identity_missing_profile"])]
            return

        if "LEFT JOIN" in query and " AS c " in query:
            if "raw_fingerprints AS c" in query:
                table_name = next(name for name in self.state["orphans"] if name.endswith("raw_fingerprints"))
            elif "feature_vectors AS c" in query:
                table_name = next(name for name in self.state["orphans"] if name.endswith("feature_vectors"))
            else:  # pragma: no cover - defensive branch for future query changes
                raise AssertionError(f"Unexpected orphan query: {query}")
            self._rows = [{"random_id": rid} for rid in list(self.state["orphans"][table_name])]
            return

        anti_join_count_match = _ANTI_JOIN_COUNT_RE.search(query)
        if anti_join_count_match:
            source_table = anti_join_count_match.group(1)
            reference_table = anti_join_count_match.group(3)
            missing_ids = self._missing_random_ids(source_table, reference_table)
            self._rows = [{"n": len(missing_ids)}]
            return

        anti_join_sample_match = _ANTI_JOIN_SAMPLE_RE.search(query)
        if anti_join_sample_match:
            source_table = anti_join_sample_match.group(2)
            reference_table = anti_join_sample_match.group(4)
            limit = int(params[0]) if params else 10
            missing_ids = self._missing_random_ids(source_table, reference_table)
            self._rows = [{"random_id": rid} for rid in missing_ids[:limit]]
            return

        delete_anti_join_match = _DELETE_ANTI_JOIN_RE.search(query)
        if delete_anti_join_match:
            source_table = delete_anti_join_match.group(1)
            reference_table = delete_anti_join_match.group(3)
            missing_ids = self._missing_random_ids(source_table, reference_table)
            if source_table in self.state["orphans"]:
                self.state["orphans"][source_table] = []
            if source_table in self.state["counts"] and self.state["counts"][source_table] is not None:
                self.state["counts"][source_table] = max(
                    0,
                    int(self.state["counts"][source_table]) - len(missing_ids),
                )
            if source_table in self.state["ids"]:
                self.state["ids"][source_table] = [
                    rid for rid in list(self.state["ids"][source_table]) if rid not in set(missing_ids)
                ]
            self.rowcount = len(missing_ids)
            self._rows = []
            return

        select_any_match = _SELECT_ANY_RE.search(query)
        if select_any_match:
            table_name = select_any_match.group(1)
            requested = {str(value) for value in list(params[0])}
            self._rows = [
                {"random_id": rid}
                for rid in list(self.state["ids"].get(table_name, []))
                if rid in requested
            ]
            return

        delete_any_match = _DELETE_ANY_RE.search(query)
        if delete_any_match:
            table_name = delete_any_match.group(1)
            delete_ids = {str(value) for value in list(params[0])}
            current_ids = list(self.state["ids"].get(table_name, []))
            remaining_ids = [rid for rid in current_ids if rid not in delete_ids]
            self.rowcount = len(current_ids) - len(remaining_ids)
            self.state["ids"][table_name] = remaining_ids
            if table_name in self.state["counts"] and self.state["counts"][table_name] is not None:
                self.state["counts"][table_name] = len(remaining_ids)
            self._rows = []
            return

        count_match = _COUNT_SQL_RE.search(query)
        if count_match:
            table_name = count_match.group(1)
            count = self.state["counts"].get(table_name)
            if count is None:
                raise AssertionError(f"Count requested for unavailable table {table_name}")
            self._rows = [{"n": int(count)}]
            return

        ids_match = _IDS_SQL_RE.search(query)
        if ids_match:
            table_name = ids_match.group(1)
            self._rows = [{"random_id": rid} for rid in list(self.state["ids"].get(table_name, []))]
            return

        raise AssertionError(f"Unexpected inspection query: {query}")

    def fetchone(self):
        if self._offset >= len(self._rows):
            return None
        row = self._rows[self._offset]
        self._offset += 1
        return row

    def fetchall(self):
        rows = list(self._rows[self._offset :])
        self._offset = len(self._rows)
        return rows

    def fetchmany(self, size: int = 1):
        rows = list(self._rows[self._offset : self._offset + int(size)])
        self._offset += len(rows)
        return rows

    def _missing_random_ids(self, source_table: str, reference_table: str) -> list[str]:
        if source_table in self.state["orphans"]:
            return list(self.state["orphans"][source_table])
        source_ids = set(self.state["ids"].get(source_table, []))
        reference_ids = set(self.state["ids"].get(reference_table, []))
        return sorted(source_ids - reference_ids)


class FakeInspectionConnection:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self, *args, **kwargs):
        _ = (args, kwargs)
        return FakeInspectionCursor(self.state)

    def commit(self):
        return None

    def close(self):
        return None


def test_for_inspection_skips_bootstrap_and_preserves_single_db_fallback(monkeypatch) -> None:
    def _boom(self) -> None:  # pragma: no cover - exercised by absence
        raise AssertionError("_init_all should not run in inspection mode")

    monkeypatch.setattr(SecureSplitFingerprintStore, "_init_all", _boom)

    store = SecureSplitFingerprintStore.for_inspection(
        database_url="postgresql://admin:secret@localhost:5432/biometric_db",
        table_prefix="demo",
    )

    layout = store.dump_layout()
    assert store.dual_database_enabled is False
    assert layout["person_table"] == "biometric_db.demo_person_directory"
    assert layout["identity_map_table"] == "biometric_db.demo_identity_map"
    assert layout["biometric_database_url"].startswith("postgresql://admin:***@")
    assert layout["identity_database_url"] == layout["biometric_database_url"]


def test_inspection_state_uses_read_only_connection_path(monkeypatch) -> None:
    biometric_url = "postgresql://admin:bio_secret@localhost:5432/biometric_db"
    identity_url = "postgresql://admin:id_secret@localhost:5433/identity_db"
    biometric_state = _inspection_state("demo_", identity_table_present=False)
    identity_state = _inspection_state("demo_", include_biometric_tables=False)
    calls: list[tuple[str, str]] = []

    def _fake_connect(self, database_url: str, *, database_role: str):
        calls.append((database_role, database_url))
        if database_url == biometric_url:
            return FakeInspectionConnection(biometric_state)
        if database_url == identity_url:
            return FakeInspectionConnection(identity_state)
        raise AssertionError(f"Unexpected database URL: {database_url}")

    monkeypatch.setattr(
        SecureSplitFingerprintStore,
        "_connect_postgres",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mutating connection path should not be used")),
    )
    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)

    payload = SecureSplitFingerprintStore.inspect_runtime_state(
        database_url=biometric_url,
        identity_database_url=identity_url,
        table_prefix="demo",
    )

    assert calls == [
        ("biometric_db", biometric_url),
        ("identity_db", identity_url),
        ("biometric_db", biometric_url),
        ("identity_db", identity_url),
    ]
    assert payload["overall_ok"] is True
    assert payload["dual_database_enabled"] is True
    assert payload["resolved_table_names"]["person"] == "biometric_db.demo_person_directory"
    assert payload["resolved_table_names"]["identity"] == "identity_db.demo_identity_map"
    assert payload["redacted_database_urls"]["biometric_db"] == "postgresql://admin:***@localhost:5432/biometric_db"
    assert payload["redacted_database_urls"]["identity_db"] == "postgresql://admin:***@localhost:5433/identity_db"
    assert payload["table_presence"]["biometric_db"]["vectors"] is True
    assert payload["vector_extension_present_in_biometric_db"] is True
    assert payload["row_counts"]["people"] == 2
    assert payload["row_counts"]["identity"] == 2
    assert payload["row_counts"]["raw"] == 2
    assert payload["row_counts"]["vectors_by_method"] == {"dl": 2, "vit": 1}
    assert payload["schema_hardening"]["identity_map_guarantees"]["contract_enforced"] is True
    assert payload["schema_hardening"]["drift"]["missing_constraints"] == []
    assert payload["schema_hardening"]["drift"]["missing_indexes"] == []
    assert payload["reconciliation"]["validation_mode"] == "bounded_exact_streaming"
    assert payload["reconciliation"]["drift_counts"]["people_without_identity_rows"] == 0
    assert payload["reconciliation"]["drift_counts"]["identity_rows_without_people"] == 0
    assert payload["reconciliation"]["repairability_summary"] == {
        "safely_repairable_issue_count": 0,
        "not_safely_repairable_issue_count": 0,
        "safely_repairable_issue_codes": [],
        "not_safely_repairable_issue_codes": [],
    }
    assert payload["errors"] == []
    assert payload["warnings"] == []
    assert payload["readiness"] == {
        "ready": True,
        "status": "ready",
        "error_count": 0,
        "warning_count": 0,
    }
    assert payload["issues"] == []


def test_inspection_state_keeps_warning_only_layouts_ready(monkeypatch) -> None:
    biometric_url = "postgresql://admin:bio_secret@localhost:5432/biometric_db"
    state = _inspection_state("warn_")
    state["identity_missing_profile"] = ["rid_b"]
    vector_table = next(name for name in state["orphans"] if name.endswith("feature_vectors"))
    state["orphans"][vector_table] = ["rid_a"]

    def _fake_connect(self, database_url: str, *, database_role: str):
        assert database_url == biometric_url
        assert database_role == "biometric_db"
        return FakeInspectionConnection(state)

    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)

    payload = SecureSplitFingerprintStore.inspect_runtime_state(
        database_url=biometric_url,
        table_prefix="warn",
    )

    assert payload["overall_ok"] is True
    assert payload["errors"] == []
    assert len(payload["warnings"]) == 2
    assert payload["integrity_warnings"] == [issue["message"] for issue in payload["warnings"]]
    assert payload["readiness"] == {
        "ready": True,
        "status": "ready_with_warnings",
        "error_count": 0,
        "warning_count": 2,
    }
    assert payload["reconciliation"]["validation_mode"] == "bounded_sql_antijoin"
    assert {issue["code"] for issue in payload["warnings"]} == {
        "identity_rows_missing_profile",
        "vector_rows_without_person",
    }
    warning_by_code = {issue["code"]: issue for issue in payload["warnings"]}
    assert warning_by_code["vector_rows_without_person"]["repairability"] == "safely_repairable"
    assert warning_by_code["identity_rows_missing_profile"]["repairability"] == "not_safely_repairable"


def test_inspection_state_marks_missing_required_tables_as_not_ready(monkeypatch) -> None:
    biometric_url = "postgresql://admin:bio_secret@localhost:5432/biometric_db"
    identity_url = "postgresql://admin:id_secret@localhost:5433/identity_db"
    biometric_state = _inspection_state("audit_", identity_table_present=False)
    identity_state = _inspection_state("audit_", identity_table_present=False, include_biometric_tables=False)

    def _fake_connect(self, database_url: str, *, database_role: str):
        if database_url == biometric_url:
            return FakeInspectionConnection(biometric_state)
        if database_url == identity_url:
            return FakeInspectionConnection(identity_state)
        raise AssertionError(f"Unexpected database URL: {database_url}")

    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)

    payload = SecureSplitFingerprintStore.inspect_runtime_state(
        database_url=biometric_url,
        identity_database_url=identity_url,
        table_prefix="audit",
    )

    assert payload["overall_ok"] is False
    assert payload["warnings"] == []
    assert payload["errors"]
    assert payload["readiness"] == {
        "ready": False,
        "status": "not_ready",
        "error_count": len(payload["errors"]),
        "warning_count": 0,
    }
    assert {issue["code"] for issue in payload["errors"]} == {"missing_table"}


def test_inspection_state_reports_schema_hardening_drift(monkeypatch) -> None:
    biometric_url = "postgresql://admin:bio_secret@localhost:5432/biometric_db"
    state = _inspection_state("drift_")
    identity_table = "drift_identity_map"
    state["indexes"]["drift_idx_person_created_at_desc"] = False
    state["constraints"][(identity_table, "drift_ck_identity_name_norm_matches_full_name")] = False

    def _fake_connect(self, database_url: str, *, database_role: str):
        assert database_url == biometric_url
        assert database_role == "biometric_db"
        return FakeInspectionConnection(state)

    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)

    payload = SecureSplitFingerprintStore.inspect_runtime_state(
        database_url=biometric_url,
        table_prefix="drift",
    )

    assert payload["overall_ok"] is False
    assert payload["schema_hardening"]["drift"]["schema_drift_detected"] is True
    assert "person_created_at_desc" in payload["schema_hardening"]["drift"]["missing_indexes"]
    assert "identity_name_norm_matches_full_name_check" in payload["schema_hardening"]["drift"]["missing_constraints"]
    assert {issue["code"] for issue in payload["errors"]} == {"identity_schema_contract_missing"}
    assert {issue["code"] for issue in payload["warnings"]} == {"supporting_indexes_missing"}


def test_inspection_state_reports_bounded_dual_database_mismatch_samples(monkeypatch) -> None:
    biometric_url = "postgresql://admin:bio_secret@localhost:5432/biometric_db"
    identity_url = "postgresql://admin:id_secret@localhost:5433/identity_db"
    biometric_state = _inspection_state(
        "drift_",
        identity_table_present=False,
        person_ids=["rid_a", "rid_b", "rid_c"],
    )
    identity_state = _inspection_state(
        "drift_",
        include_biometric_tables=False,
        identity_ids=["rid_a", "rid_b", "rid_orphan"],
    )

    def _fake_connect(self, database_url: str, *, database_role: str):
        if database_url == biometric_url:
            return FakeInspectionConnection(biometric_state)
        if database_url == identity_url:
            return FakeInspectionConnection(identity_state)
        raise AssertionError(f"Unexpected database URL: {database_url}")

    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)

    payload = SecureSplitFingerprintStore.inspect_runtime_state(
        database_url=biometric_url,
        identity_database_url=identity_url,
        table_prefix="drift",
    )

    warnings_by_code = {issue["code"]: issue for issue in payload["warnings"]}
    assert payload["reconciliation"]["validation_mode"] == "bounded_exact_streaming"
    assert payload["reconciliation"]["drift_counts"]["people_without_identity_rows"] == 1
    assert payload["reconciliation"]["drift_counts"]["identity_rows_without_people"] == 1
    assert warnings_by_code["people_without_identity_rows"]["sample_random_ids"] == ["rid_c"]
    assert warnings_by_code["identity_rows_without_people"]["sample_random_ids"] == ["rid_orphan"]
    assert warnings_by_code["identity_rows_without_people"]["repairability"] == "safely_repairable"
    assert "repair-identity-orphans" in warnings_by_code["identity_rows_without_people"]["remediation"]


def test_show_script_uses_single_db_fallback_and_redacts_urls(monkeypatch, capsys) -> None:
    module = _load_script_module("show_identification_runtime_db.py")
    biometric_url = "postgresql://admin:single_secret@localhost:5432/biometric_db"
    state = _inspection_state("browser_")
    calls: list[tuple[str, str]] = []

    def _fake_connect(self, database_url: str, *, database_role: str):
        calls.append((database_role, database_url))
        assert database_url == biometric_url
        return FakeInspectionConnection(state)

    monkeypatch.setenv("IDENTIFICATION_TEST_BIOMETRIC_DATABASE_URL", biometric_url)
    monkeypatch.delenv("IDENTIFICATION_TEST_IDENTITY_DATABASE_URL", raising=False)
    monkeypatch.delenv("IDENTITY_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("IDENTIFICATION_TABLE_PREFIX", "browser")
    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)

    exit_code = module.main([])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls == [
        ("biometric_db", biometric_url),
        ("biometric_db", biometric_url),
    ]
    assert payload["dual_database_enabled"] is False
    assert payload["redacted_database_urls"]["identity_db"] == payload["redacted_database_urls"]["biometric_db"]
    assert payload["resolved_table_names"]["person"] == "biometric_db.browser_person_directory"
    assert "single_secret" not in json.dumps(payload)


def test_check_script_fails_when_identity_table_is_missing(monkeypatch, capsys) -> None:
    module = _load_script_module("check_identification_db_schema.py")
    biometric_url = "postgresql://admin:bio_secret@localhost:5432/biometric_db"
    identity_url = "postgresql://admin:id_secret@localhost:5433/identity_db"
    biometric_state = _inspection_state("audit_", identity_table_present=False)
    identity_state = _inspection_state("audit_", identity_table_present=False, include_biometric_tables=False)

    def _fake_connect(self, database_url: str, *, database_role: str):
        if database_url == biometric_url:
            return FakeInspectionConnection(biometric_state)
        if database_url == identity_url:
            return FakeInspectionConnection(identity_state)
        raise AssertionError(f"Unexpected database URL: {database_url}")

    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)

    exit_code = module.main(
        [
            "--database-url",
            biometric_url,
            "--identity-database-url",
            identity_url,
            "--table-prefix",
            "audit",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "missing_table" in output
    assert "bio_secret" not in output
    assert "id_secret" not in output


def test_check_script_allows_warning_only_layouts(monkeypatch, capsys) -> None:
    module = _load_script_module("check_identification_db_schema.py")
    biometric_url = "postgresql://admin:warn_secret@localhost:5432/biometric_db"
    state = _inspection_state("warn_")
    state["identity_missing_profile"] = ["rid_b"]

    def _fake_connect(self, database_url: str, *, database_role: str):
        assert database_url == biometric_url
        assert database_role == "biometric_db"
        return FakeInspectionConnection(state)

    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)

    exit_code = module.main(
        [
            "--database-url",
            biometric_url,
            "--table-prefix",
            "warn",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "ready_with_warnings" in output
    assert "warn_secret" not in output


def test_reconcile_script_report_only_outputs_structured_report(monkeypatch, capsys) -> None:
    module = _load_script_module("reconcile_identification_runtime_db.py")
    biometric_url = "postgresql://admin:bio_secret@localhost:5432/biometric_db"
    identity_url = "postgresql://admin:id_secret@localhost:5433/identity_db"
    biometric_state = _inspection_state(
        "recon_",
        identity_table_present=False,
        person_ids=["rid_a", "rid_b", "rid_c"],
    )
    identity_state = _inspection_state(
        "recon_",
        include_biometric_tables=False,
        identity_ids=["rid_a", "rid_b"],
    )

    def _fake_connect(self, database_url: str, *, database_role: str):
        if database_url == biometric_url:
            return FakeInspectionConnection(biometric_state)
        if database_url == identity_url:
            return FakeInspectionConnection(identity_state)
        raise AssertionError(f"Unexpected database URL: {database_url}")

    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)

    exit_code = module.main(
        [
            "--database-url",
            biometric_url,
            "--identity-database-url",
            identity_url,
            "--table-prefix",
            "recon",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["report_mode"] == "report_only"
    assert report["summary"]["severity"] == {
        "informational": 0,
        "warning": 1,
        "error": 0,
    }
    assert report["summary"]["repairability"] == {
        "safely_repairable": 0,
        "not_safely_repairable": 1,
    }
    assert report["issues"][0]["code"] == "people_without_identity_rows"
    assert "cannot be reconstructed safely" in report["issues"][0]["remediation"]
    assert report["commands"]["report_only"].endswith("--table-prefix recon_")


def test_reconcile_script_can_apply_safe_raw_orphan_repair(monkeypatch, capsys) -> None:
    module = _load_script_module("reconcile_identification_runtime_db.py")
    biometric_url = "postgresql://admin:single_secret@localhost:5432/biometric_db"
    state = _inspection_state("repair_", raw_orphan_ids=["rid_orphan"])

    def _fake_connect(self, database_url: str, *, database_role: str):
        assert database_url == biometric_url
        assert database_role == "biometric_db"
        return FakeInspectionConnection(state)

    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)
    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_biometric", lambda self: FakeInspectionConnection(state))

    exit_code = module.main(
        [
            "--database-url",
            biometric_url,
            "--table-prefix",
            "repair",
            "--repair-raw-orphans",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["report_mode"] == "repair"
    assert report["applied_repairs"][0]["action"] == "repair_raw_orphans"
    assert report["applied_repairs"][0]["deleted_rows"] == 1
    assert {issue["code"] for issue in report["inspection"]["issues"]} == set()


def test_reconcile_script_can_apply_explicit_identity_orphan_repair(monkeypatch, capsys) -> None:
    module = _load_script_module("reconcile_identification_runtime_db.py")
    biometric_url = "postgresql://admin:bio_secret@localhost:5432/biometric_db"
    identity_url = "postgresql://admin:id_secret@localhost:5433/identity_db"
    biometric_state = _inspection_state(
        "repairid_",
        identity_table_present=False,
        person_ids=["rid_a", "rid_b"],
    )
    identity_state = _inspection_state(
        "repairid_",
        include_biometric_tables=False,
        identity_ids=["rid_a", "rid_b", "rid_orphan"],
    )

    def _fake_connect(self, database_url: str, *, database_role: str):
        if database_url == biometric_url:
            return FakeInspectionConnection(biometric_state)
        if database_url == identity_url:
            return FakeInspectionConnection(identity_state)
        raise AssertionError(f"Unexpected database URL: {database_url}")

    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_postgres_inspection", _fake_connect)
    monkeypatch.setattr(SecureSplitFingerprintStore, "_connect_identity", lambda self: FakeInspectionConnection(identity_state))

    exit_code = module.main(
        [
            "--database-url",
            biometric_url,
            "--identity-database-url",
            identity_url,
            "--table-prefix",
            "repairid",
            "--repair-identity-orphans",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["applied_repairs"][0]["action"] == "repair_identity_orphans"
    assert report["applied_repairs"][0]["deleted_rows"] == 1
    assert "rid_orphan" in report["applied_repairs"][0]["deleted_sample_random_ids"]
    assert {issue["code"] for issue in report["inspection"]["issues"]} == set()
