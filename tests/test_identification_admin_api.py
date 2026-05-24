from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import apps.api.main as api_main
from apps.api.schemas import (
    IdentificationAdminInspectionResponse,
    IdentificationAdminReconciliationResponse,
)
import apps.api.service as api_service
import src.fpbench.identification.secure_split_store as secure_store_module


DIRECT_RETRIEVAL_METHODS = ["classic_orb", "classic_gftt_orb", "minutiae", "harris", "sift", "dl", "vit"]
RERANK_ONLY_METHODS = ["sift_plain_roll_v2", "dedicated"]


class _FakeMatchService:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def method_availability(self) -> dict[str, dict[str, object]]:
        return {}


class _FakeStore:
    def total_people(self) -> int:
        return 0

    def dump_layout(self) -> dict[str, str]:
        return {
            "backend": "memory",
            "dual_database_enabled": "false",
            "person_table": "memory.person_directory",
            "raw_fingerprints_table": "memory.raw_fingerprints",
            "feature_vectors_table": "memory.feature_vectors",
            "identity_map_table": "memory.identity_map",
        }

    def get_person(self, random_id: str):
        return None

    def purge(self, random_id: str) -> bool:
        return False


class _FakeIdentificationService:
    created_prefixes: list[str] = []

    def __init__(
        self,
        *,
        database_url: str | None = None,
        identity_database_url: str | None = None,
        table_prefix: str = "",
        match_service=None,
        **_: object,
    ) -> None:
        type(self).created_prefixes.append(table_prefix)
        self.database_url = database_url
        self.identity_database_url = identity_database_url
        self.table_prefix = table_prefix
        self.match_service = match_service
        self.store = _FakeStore()

    def stats(self) -> dict[str, object]:
        return {
            "total_enrolled": self.store.total_people(),
            "storage_layout": self.store.dump_layout(),
        }


class _BrokenMatchService:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("match bootstrap exploded")


class _BrokenIdentificationService(_FakeIdentificationService):
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("operational identification bootstrap exploded")


class _BrowserInitFailingIdentificationService(_FakeIdentificationService):
    def __init__(self, *args, table_prefix: str = "", **kwargs) -> None:
        if table_prefix == "identify_browser_":
            raise RuntimeError("browser identification bootstrap exploded")
        super().__init__(*args, table_prefix=table_prefix, **kwargs)


def _inspection_payload(
    *,
    overall_ok: bool = True,
    readiness_status: str = "ready",
    errors: list[dict[str, object]] | None = None,
    warnings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    error_items = list(errors or [])
    warning_items = list(warnings or [])
    method_capabilities = api_main.SecureSplitFingerprintStore.method_capability_metadata()
    return {
        "backend": "postgresql",
        "layout_version": "v4_dual_database_identity_profile_split",
        "dual_database_enabled": True,
        "table_prefix": "",
        "redacted_database_urls": {
            "biometric_db": "postgresql://admin:***@localhost:5432/biometric_db",
            "identity_db": "postgresql://admin:***@localhost:5433/identity_db",
        },
        "resolved_table_names": {
            "person": "biometric_db.person_directory",
            "identity": "identity_db.identity_map",
            "raw": "biometric_db.raw_fingerprints",
            "vectors": "biometric_db.feature_vectors",
            "generic_vectors": "biometric_db.method_retrieval_vectors",
        },
        "table_presence": {
            "biometric_db": {
                "person": True,
                "identity": False,
                "raw": True,
                "vectors": True,
                "generic_vectors": True,
            },
            "identity_db": {
                "person": False,
                "identity": True,
                "raw": False,
                "vectors": False,
                "generic_vectors": False,
            },
        },
        "row_counts": {
            "people": 4,
            "identity": 4,
            "raw": 4,
            "vectors_by_method": {"dl": 4, "vit": 4},
            "legacy_vectors_by_method": {"dl": 4, "vit": 4},
            "generic_vectors_by_method_kind": {
                "dl/deep_embedding_resnet": 4,
                "vit/vit_embedding": 4,
            },
        },
        "vector_extension_present_in_biometric_db": True,
        "unexpected_vector_methods": {},
        "method_capabilities": method_capabilities,
        "retrieval_capabilities": method_capabilities,
        "direct_vector_retrieval_methods": DIRECT_RETRIEVAL_METHODS,
        "rerank_only_methods": RERANK_ONLY_METHODS,
        "vector_storage_schema": api_main.SecureSplitFingerprintStore.vector_storage_schema_metadata(),
        "schema_hardening": {
            "identity_map_guarantees": {
                "contract_enforced": overall_ok,
                "profiles_complete": True,
                "completeness_guaranteed": overall_ok,
            },
            "drift": {
                "schema_drift_detected": not overall_ok,
                "missing_indexes": [],
                "missing_constraints": [],
                "legacy_schema_elements": [],
            },
        },
        "template_protection": {
            "raw_image_storage_policy": "metadata_only_new_writes",
            "new_raw_image_persistence_enabled": False,
            "raw_image_bytes_column_present": True,
            "raw_image_bytes_column_nullable": True,
            "legacy_raw_image_bytes_row_count": 0,
            "legacy_raw_image_bytes_sample": [],
            "legacy_raw_image_storage_status": "clear",
            "rerank_legacy_bytes_adapter_enabled": True,
        },
        "integrity_warnings": [str(item["message"]) for item in warning_items],
        "overall_ok": overall_ok,
        "readiness": {
            "ready": overall_ok,
            "status": readiness_status,
            "error_count": len(error_items),
            "warning_count": len(warning_items),
        },
        "errors": error_items,
        "warnings": warning_items,
        "issues": [*error_items, *warning_items],
    }


def _reconciliation_payload(
    *,
    inspection: dict[str, object] | None = None,
) -> dict[str, object]:
    inspection_payload = inspection or _inspection_payload()
    readiness = dict(inspection_payload["readiness"])
    issues = list(inspection_payload["issues"])
    return {
        "generated_at": "2026-04-22T09:30:00+00:00",
        "report_mode": "report_only",
        "requested_repairs": [],
        "available_repairs": ["repair_identity_orphans"] if issues else [],
        "applied_repairs": [],
        "summary": {
            "severity": {
                "informational": 0,
                "warning": readiness["warning_count"],
                "error": readiness["error_count"],
            },
            "repairability": {
                "safely_repairable": 0,
                "not_safely_repairable": len(issues),
            },
            "manual_reconciliation_required": False,
            "overall_ok": inspection_payload["overall_ok"],
            "readiness": readiness,
        },
        "commands": {
            "report_only": "python scripts/diagnostics/reconcile_identification_runtime_db.py",
            "repair_raw_orphans": (
                "python scripts/diagnostics/reconcile_identification_runtime_db.py --repair-raw-orphans"
            ),
            "repair_vector_orphans": (
                "python scripts/diagnostics/reconcile_identification_runtime_db.py --repair-vector-orphans"
            ),
            "repair_identity_orphans": (
                "python scripts/diagnostics/reconcile_identification_runtime_db.py --repair-identity-orphans"
            ),
            "redact_legacy_raw_image_bytes": (
                "python scripts/diagnostics/reconcile_identification_runtime_db.py --redact-legacy-raw-image-bytes"
            ),
            "backfill_generic_retrieval_vectors": (
                "python scripts/diagnostics/reconcile_identification_runtime_db.py --backfill-generic-retrieval-vectors"
            ),
        },
        "inspection": inspection_payload,
        "issues": issues,
    }


def test_admin_inspection_schema_accepts_template_protection_section() -> None:
    payload = _inspection_payload()
    response = IdentificationAdminInspectionResponse(**payload)

    assert response.template_protection["raw_image_storage_policy"] == "metadata_only_new_writes"
    assert response.template_protection["legacy_raw_image_storage_status"] == "clear"
    assert response.template_protection["legacy_raw_image_bytes_row_count"] == 0


def test_admin_response_models_accept_full_direct_retrieval_contract() -> None:
    inspection = IdentificationAdminInspectionResponse(**_inspection_payload())
    reconciliation = IdentificationAdminReconciliationResponse(
        **_reconciliation_payload(inspection=_inspection_payload())
    )

    assert inspection.direct_vector_retrieval_methods == DIRECT_RETRIEVAL_METHODS
    assert reconciliation.inspection.direct_vector_retrieval_methods == DIRECT_RETRIEVAL_METHODS
    assert "dedicated" not in inspection.direct_vector_retrieval_methods


def test_admin_reconciliation_schema_accepts_legacy_redaction_repair_result() -> None:
    inspection = _inspection_payload(
        readiness_status="ready_with_warnings",
        warnings=[
            {
                "code": "legacy_raw_image_bytes_present",
                "severity": "warning",
                "database_role": "biometric_db",
                "message": "legacy raw image bytes are present.",
                "repair_actions": ["redact_legacy_raw_image_bytes"],
            }
        ],
    )
    payload = _reconciliation_payload(inspection=inspection)
    payload["available_repairs"] = ["redact_legacy_raw_image_bytes"]
    payload["applied_repairs"] = [
        {
            "action": "redact_legacy_raw_image_bytes",
            "redacted_count": 2,
            "table": "raw_fingerprints",
            "column": "image_bytes",
            "sensitive_backup_created": False,
        }
    ]

    response = IdentificationAdminReconciliationResponse(**payload)

    assert response.available_repairs == ["redact_legacy_raw_image_bytes"]
    assert response.applied_repairs[0]["redacted_count"] == 2
    assert response.applied_repairs[0]["sensitive_backup_created"] is False


@pytest.fixture(autouse=True)
def _restore_api_globals(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    original_service = api_main._service
    original_service_init_error = api_main._service_init_error
    original_ident_service = api_main._ident_service
    original_ident_service_init_error = api_main._ident_service_init_error
    original_browser_service = api_main._browser_ident_service
    original_browser_service_init_error = api_main._browser_ident_service_init_error

    _FakeIdentificationService.created_prefixes = []
    api_main._service = None
    api_main._service_init_error = None
    api_main._ident_service = None
    api_main._ident_service_init_error = None
    api_main._browser_ident_service = None
    api_main._browser_ident_service_init_error = None

    monkeypatch.setenv("FPBENCH_IDENTIFY_BROWSER_STATE_PATH", str(tmp_path / "identify_browser_state.json"))
    yield

    api_main._service = original_service
    api_main._service_init_error = original_service_init_error
    api_main._ident_service = original_ident_service
    api_main._ident_service_init_error = original_ident_service_init_error
    api_main._browser_ident_service = original_browser_service
    api_main._browser_ident_service_init_error = original_browser_service_init_error


def test_collect_identification_admin_state_uses_store_inspection_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_inspect_runtime_state(cls, **kwargs):
        calls.append(dict(kwargs))
        return _inspection_payload()

    monkeypatch.setattr(
        api_main.SecureSplitFingerprintStore,
        "inspect_runtime_state",
        classmethod(_fake_inspect_runtime_state),
    )

    payload = api_main._collect_identification_admin_state(store_scope="browser")

    assert payload["backend"] == "postgresql"
    assert calls == [
        {
            "database_url": None,
            "identity_database_url": None,
            "table_prefix": "identify_browser_",
        }
    ]


def test_collect_identification_admin_reconciliation_report_uses_store_reconciliation_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_reconcile_runtime_state(cls, **kwargs):
        calls.append(dict(kwargs))
        return _reconciliation_payload()

    monkeypatch.setattr(
        api_main.SecureSplitFingerprintStore,
        "reconcile_runtime_state",
        classmethod(_fake_reconcile_runtime_state),
    )

    payload = api_main._collect_identification_admin_reconciliation_report(store_scope="browser")

    assert payload["report_mode"] == "report_only"
    assert calls == [
        {
            "database_url": None,
            "identity_database_url": None,
            "table_prefix": "identify_browser_",
        }
    ]


def test_admin_layout_endpoint_returns_redacted_read_only_inspection_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect_calls: list[dict[str, object]] = []

    def _fake_inspect_runtime_state(cls, **kwargs):
        inspect_calls.append(dict(kwargs))
        return _inspection_payload()

    def _unexpected_reconcile_runtime_state(cls, **kwargs):
        raise AssertionError("layout endpoint should stay on the inspection contract")

    monkeypatch.setattr(api_service, "MatchService", _FakeMatchService)
    monkeypatch.setattr(api_main, "IdentificationService", _FakeIdentificationService)
    monkeypatch.setattr(
        api_main.SecureSplitFingerprintStore,
        "inspect_runtime_state",
        classmethod(_fake_inspect_runtime_state),
    )
    monkeypatch.setattr(
        api_main.SecureSplitFingerprintStore,
        "reconcile_runtime_state",
        classmethod(_unexpected_reconcile_runtime_state),
    )

    with TestClient(api_main.app) as client:
        response = client.get("/identify/admin/layout?store_scope=browser")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "postgresql"
    assert payload["overall_ok"] is True
    assert payload["schema_hardening"]["identity_map_guarantees"]["contract_enforced"] is True
    assert payload["template_protection"]["raw_image_storage_policy"] == "metadata_only_new_writes"
    assert payload["template_protection"]["legacy_raw_image_storage_status"] == "clear"
    assert payload["direct_vector_retrieval_methods"] == DIRECT_RETRIEVAL_METHODS
    assert payload["rerank_only_methods"] == RERANK_ONLY_METHODS
    assert payload["method_capabilities"]["dl"]["retrieval_vector_dim"] == 512
    assert payload["method_capabilities"]["sift"]["retrieval_vector_dim"] == 512
    assert payload["method_capabilities"]["sift"]["retrieval_vector_kind"] == "sift_aggregated_descriptor_v1"
    dedicated_capability = payload["method_capabilities"]["dedicated"]
    assert dedicated_capability["retrieval_unavailable_reason"] == (
        "experimental_rerank_only_no_validated_global_retrieval_vector_yet"
    )
    assert dedicated_capability["retrieval_capability_status"] == "experimental_rerank_only"
    assert dedicated_capability["direct_retrieval_exclusion"] == "intentional_rerank_only"
    assert dedicated_capability["experimental"] is True
    assert dedicated_capability["supports_direct_vector_retrieval"] is False
    assert dedicated_capability["supports_pairwise_rerank"] is True
    assert dedicated_capability["future_adapter_hint"] == (
        "A future dedicated_aggregated_patch_descriptor_v1 adapter can be added once global pooling is validated."
    )
    assert payload["vector_storage_schema"]["method_generic_vectors_supported"] is True
    assert payload["vector_storage_schema"]["schema_accepts_method_generic_vectors"] is True
    assert payload["vector_storage_schema"]["legacy_compatibility_methods"] == ["dl", "vit"]
    assert payload["vector_storage_schema"]["dual_write_methods"] == ["dl", "vit"]
    assert payload["vector_storage_schema"]["generic_only_methods"] == [
        "classic_gftt_orb",
        "classic_orb",
        "harris",
        "minutiae",
        "sift",
    ]
    assert payload["redacted_database_urls"]["biometric_db"] == "postgresql://admin:***@localhost:5432/biometric_db"
    assert "localhost:5433/identity_db" in payload["redacted_database_urls"]["identity_db"]
    assert "secret" not in json.dumps(payload)
    assert inspect_calls == [
        {
            "database_url": None,
            "identity_database_url": None,
            "table_prefix": "identify_browser_",
        }
    ]
    assert api_main._browser_ident_service is None
    assert _FakeIdentificationService.created_prefixes == [""]


def test_lifespan_startup_initializes_operational_services_and_shutdown_clears_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_service, "MatchService", _FakeMatchService)
    monkeypatch.setattr(api_main, "IdentificationService", _FakeIdentificationService)

    assert api_main.app.router.on_startup == []

    with TestClient(api_main.app) as client:
        health = client.get("/health")

        assert health.status_code == 200
        assert api_main._service is not None
        assert api_main._ident_service is not None
        assert api_main._browser_ident_service is None
        assert health.json()["status"] == "ready"
        assert health.json()["identify_status"] == "ready"
        assert health.json()["identify_browser_status"] == "lazy_not_initialized"
        assert health.json()["direct_vector_retrieval_methods"] == DIRECT_RETRIEVAL_METHODS
        assert "dedicated" not in health.json()["direct_vector_retrieval_methods"]

    assert api_main._service is None
    assert api_main._ident_service is None
    assert api_main._browser_ident_service is None


def test_admin_reconcile_endpoint_returns_reconciliation_report_without_initializing_browser_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconcile_calls: list[dict[str, object]] = []
    warning = {
        "code": "identity_rows_missing_profile",
        "severity": "warning",
        "database_role": "identity_db",
        "message": "identity_map contains rows missing full_name/name_norm.",
    }
    error = {
        "code": "missing_table",
        "severity": "error",
        "database_role": "identity_db",
        "message": "identity_db is missing expected table identity_map.",
    }
    inspection = _inspection_payload(
        overall_ok=False,
        readiness_status="not_ready",
        errors=[error],
        warnings=[warning],
    )

    def _fake_reconcile_runtime_state(cls, **kwargs):
        reconcile_calls.append(dict(kwargs))
        return _reconciliation_payload(inspection=inspection)

    def _unexpected_inspect_runtime_state(cls, **kwargs):
        raise AssertionError("reconcile endpoint should use the reconciliation report contract")

    monkeypatch.setattr(api_service, "MatchService", _FakeMatchService)
    monkeypatch.setattr(api_main, "IdentificationService", _FakeIdentificationService)
    monkeypatch.setattr(
        api_main.SecureSplitFingerprintStore,
        "reconcile_runtime_state",
        classmethod(_fake_reconcile_runtime_state),
    )
    monkeypatch.setattr(
        api_main.SecureSplitFingerprintStore,
        "inspect_runtime_state",
        classmethod(_unexpected_inspect_runtime_state),
    )

    with TestClient(api_main.app) as client:
        response = client.get("/identify/admin/reconcile?store_scope=browser")

    assert response.status_code == 200
    payload = response.json()
    assert payload["report_mode"] == "report_only"
    assert payload["requested_repairs"] == []
    assert payload["summary"]["overall_ok"] is False
    assert payload["summary"]["readiness"]["status"] == "not_ready"
    assert payload["inspection"]["errors"][0]["code"] == "missing_table"
    assert payload["inspection"]["warnings"][0]["code"] == "identity_rows_missing_profile"
    assert payload["inspection"]["schema_hardening"]["drift"]["schema_drift_detected"] is True
    assert payload["issues"][0]["code"] == "missing_table"
    assert "secret" not in json.dumps(payload)
    assert reconcile_calls == [
        {
            "database_url": None,
            "identity_database_url": None,
            "table_prefix": "identify_browser_",
        }
    ]
    assert api_main._browser_ident_service is None
    assert _FakeIdentificationService.created_prefixes == [""]


def test_browser_service_initialization_is_lazy_until_a_browser_endpoint_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_service, "MatchService", _FakeMatchService)
    monkeypatch.setattr(api_main, "IdentificationService", _FakeIdentificationService)

    with TestClient(api_main.app) as client:
        health_before = client.get("/health")

        assert health_before.status_code == 200
        assert health_before.json()["identify_browser_ok"] is True
        assert health_before.json()["identify_browser_initialized"] is False
        assert health_before.json()["identify_browser_error"] is None
        assert health_before.json()["identify_browser_status"] == "lazy_not_initialized"
        assert _FakeIdentificationService.created_prefixes == [""]

        response = client.post("/identify/browser/reset")
        health_after = client.get("/health")

    assert response.status_code == 200
    assert health_after.status_code == 200
    assert health_after.json()["identify_browser_ok"] is True
    assert health_after.json()["identify_browser_initialized"] is True
    assert health_after.json()["identify_browser_error"] is None
    assert health_after.json()["identify_browser_status"] == "initialized"
    assert _FakeIdentificationService.created_prefixes == ["", "identify_browser_"]


def test_health_surfaces_match_service_startup_failure_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_service, "MatchService", _BrokenMatchService)
    monkeypatch.setattr(api_main, "IdentificationService", _FakeIdentificationService)

    with TestClient(api_main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["status"] == "error"
    assert "match bootstrap exploded" in payload["error"]
    assert payload["identify_ok"] is False
    assert payload["identify_status"] == "blocked"
    assert "MatchService failed" in payload["identify_error"]
    assert payload["identify_browser_ok"] is False
    assert payload["identify_browser_initialized"] is False
    assert payload["identify_browser_status"] == "blocked"
    assert "MatchService failed" in payload["identify_browser_error"]


def test_health_surfaces_operational_identification_startup_failure_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_service, "MatchService", _FakeMatchService)
    monkeypatch.setattr(api_main, "IdentificationService", _BrokenIdentificationService)

    with TestClient(api_main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["status"] == "ready"
    assert payload["identify_ok"] is False
    assert payload["identify_status"] == "error"
    assert "operational identification bootstrap exploded" in payload["identify_error"]
    assert payload["identify_browser_ok"] is False
    assert payload["identify_browser_initialized"] is False
    assert payload["identify_browser_status"] == "error"
    assert "operational identification bootstrap exploded" in payload["identify_browser_error"]


def test_browser_initialization_failure_is_visible_after_lazy_init_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_service, "MatchService", _FakeMatchService)
    monkeypatch.setattr(api_main, "IdentificationService", _BrowserInitFailingIdentificationService)

    with TestClient(api_main.app) as client:
        health_before = client.get("/health")
        response = client.post("/identify/browser/reset")
        health_after = client.get("/health")

    assert health_before.status_code == 200
    assert health_before.json()["identify_browser_status"] == "lazy_not_initialized"
    assert response.status_code == 500
    assert "browser identification bootstrap exploded" in response.json()["detail"]
    assert health_after.status_code == 200
    assert health_after.json()["identify_browser_ok"] is False
    assert health_after.json()["identify_browser_initialized"] is False
    assert health_after.json()["identify_browser_status"] == "error"
    assert "browser identification bootstrap exploded" in health_after.json()["identify_browser_error"]


def _install_missing_password_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    def _connect(*args, **kwargs):
        raise RuntimeError("fe_sendauth: no password supplied")

    monkeypatch.setattr(
        secure_store_module,
        "_load_postgres_base_deps",
        lambda: (SimpleNamespace(connect=_connect), object()),
    )


def test_identify_stats_missing_database_password_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_missing_password_psycopg(monkeypatch)
    monkeypatch.setattr(api_service, "MatchService", _FakeMatchService)
    monkeypatch.setenv("FPBENCH_API_LAZY_STARTUP", "1")
    monkeypatch.setenv("DATABASE_URL", "postgresql://admin@127.0.0.1:5432/biometric_db")
    monkeypatch.delenv("IDENTITY_DATABASE_URL", raising=False)

    with TestClient(api_main.app) as client:
        response = client.get("/identify/stats")

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "fe_sendauth: no password supplied" in detail
    assert "Set DATABASE_URL" in detail
    assert "postgresql://admin:<password>@127.0.0.1:5432/biometric_db" in detail
    assert "docs/LOCAL_DUAL_DB_RUNBOOK.md" in detail
    assert "change_me_biometric_dev_password" not in detail


def test_admin_layout_missing_database_password_returns_readiness_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_missing_password_psycopg(monkeypatch)
    monkeypatch.setenv("FPBENCH_API_LAZY_STARTUP", "1")
    monkeypatch.setenv("DATABASE_URL", "postgresql://admin@127.0.0.1:5432/biometric_db")
    monkeypatch.setenv("IDENTITY_DATABASE_URL", "postgresql://admin@127.0.0.1:5433/identity_db")

    with TestClient(api_main.app) as client:
        response = client.get("/identify/admin/layout")

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_ok"] is False
    assert payload["readiness"]["status"] == "not_ready"
    assert payload["direct_vector_retrieval_methods"] == DIRECT_RETRIEVAL_METHODS
    connection_issues = [
        issue for issue in payload["issues"] if issue["code"] == "database_connection_failed"
    ]
    assert {issue["database_role"] for issue in connection_issues} == {"biometric_db", "identity_db"}
    assert any("Set DATABASE_URL" in issue["message"] for issue in connection_issues)
    assert any("Set IDENTITY_DATABASE_URL" in issue["message"] for issue in connection_issues)
    assert "change_me_biometric_dev_password" not in json.dumps(payload)
