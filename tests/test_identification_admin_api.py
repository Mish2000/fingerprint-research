from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import apps.api.main as api_main
import apps.api.service as api_service


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
        },
        "table_presence": {
            "biometric_db": {
                "person": True,
                "identity": False,
                "raw": True,
                "vectors": True,
            },
            "identity_db": {
                "person": False,
                "identity": True,
                "raw": False,
                "vectors": False,
            },
        },
        "row_counts": {
            "people": 4,
            "identity": 4,
            "raw": 4,
            "vectors_by_method": {"dl": 4, "vit": 4},
        },
        "vector_extension_present_in_biometric_db": True,
        "unexpected_vector_methods": {},
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
        },
        "inspection": inspection_payload,
        "issues": issues,
    }


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
