from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import os
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import apps.api.service as api_service
from apps.api.benchmark_catalog import (
    load_benchmark_runs,
    load_benchmark_summary,
    load_best_methods,
    load_comparison,
    resolve_benchmark_artifact,
)
from apps.api.catalog_store import (
    CatalogApiError,
    CatalogArtifactError,
    CatalogBrowserAssetNotFoundError,
    CatalogDatasetNotFoundError,
    CatalogInvalidRequestError,
    CatalogVerifyCaseNotFoundError,
    load_catalog_dataset_browser,
    load_catalog_datasets,
    load_catalog_identify_gallery,
    load_catalog_verify_case_detail,
    load_catalog_verify_cases,
    resolve_catalog_browser_asset_path,
)
from apps.api.demo_store import (
    DemoAssetResolutionError,
    DemoCaseNotFoundError,
    DemoInvalidSlotError,
    DemoStoreError,
    load_demo_cases,
    resolve_demo_case_path,
)
from apps.api.identification_service import IdentificationService
from apps.api.identify_demo_store import (
    IdentifyDemoStoreError,
    get_identify_browser_seeded_count,
    get_identify_demo_seeded_count,
    reset_identify_browser_store,
    reset_identify_demo_store,
    seed_identify_browser_store,
    seed_identify_demo_store,
)
from apps.api.io import save_upload_to_temp
from apps.api.method_registry import MethodRegistryError, load_api_method_registry
from apps.api.schemas import (
    CatalogDatasetBrowserResponse,
    CatalogDatasetsResponse,
    CatalogIdentifyGalleryResponse,
    CatalogVerifyCaseDetail,
    CatalogVerifyCasesResponse,
    DeleteIdentityResponse,
    DemoCasesResponse,
    EnrollFingerprintResponse,
    IdentificationAdminInspectionResponse,
    IdentificationAdminReconciliationResponse,
    IdentificationStatsResponse,
    IdentifyBrowserResetResponse,
    IdentifyBrowserSeedSelectionRequest,
    IdentifyBrowserSeedSelectionResponse,
    IdentifyDemoResetResponse,
    IdentifyDemoSeedResponse,
    IdentifyResponse,
    MatchMethod,
    MatchResponse,
)
from src.fpbench.identification.secure_split_store import IdentifyHints, SecureSplitFingerprintStore


StoreScope = Literal["operational", "browser"]
BROWSER_TABLE_PREFIX = "identify_browser_"

_service: api_service.MatchService | None = None
_service_init_error: str | None = None
_ident_service: IdentificationService | None = None
_ident_service_init_error: str | None = None
_browser_ident_service: IdentificationService | None = None
_browser_ident_service_init_error: str | None = None


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _initialize_match_service() -> None:
    global _service, _service_init_error
    if _service is not None or _service_init_error is not None:
        return
    try:
        _service = api_service.MatchService()
    except Exception as exc:  # pragma: no cover - exercised through health tests
        _service = None
        _service_init_error = _format_error(exc)


def _initialize_identification_service() -> None:
    global _ident_service, _ident_service_init_error
    if _ident_service is not None or _ident_service_init_error is not None:
        return

    _initialize_match_service()
    if _service is None:
        _ident_service_init_error = f"MatchService failed: {_service_init_error or 'unknown startup error'}"
        return

    try:
        _ident_service = IdentificationService(match_service=_service)
    except Exception as exc:  # pragma: no cover - exercised through health tests
        _ident_service = None
        _ident_service_init_error = _format_error(exc)


def _initialize_browser_identification_service() -> None:
    global _browser_ident_service, _browser_ident_service_init_error
    if _browser_ident_service is not None or _browser_ident_service_init_error is not None:
        return

    _initialize_match_service()
    if _service is None:
        _browser_ident_service_init_error = f"MatchService failed: {_service_init_error or 'unknown startup error'}"
        return
    if _ident_service_init_error is not None:
        _browser_ident_service_init_error = _ident_service_init_error
        return

    try:
        _browser_ident_service = IdentificationService(
            match_service=_service,
            table_prefix=BROWSER_TABLE_PREFIX,
        )
    except Exception as exc:  # pragma: no cover - exercised through health tests
        _browser_ident_service = None
        _browser_ident_service_init_error = _format_error(exc)


def _shutdown_services() -> None:
    global _service, _service_init_error
    global _ident_service, _ident_service_init_error
    global _browser_ident_service, _browser_ident_service_init_error
    _service = None
    _service_init_error = None
    _ident_service = None
    _ident_service_init_error = None
    _browser_ident_service = None
    _browser_ident_service_init_error = None


def _lazy_startup_enabled() -> bool:
    return os.getenv("FPBENCH_API_LAZY_STARTUP", "").strip().lower() in {"1", "true", "yes"}


@asynccontextmanager
async def _lifespan(_: FastAPI):
    if not _lazy_startup_enabled():
        _initialize_match_service()
        _initialize_identification_service()
    try:
        yield
    finally:
        _shutdown_services()


app = FastAPI(title="Fingerprint Research API", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
router = APIRouter()


def _get_match_service() -> api_service.MatchService:
    _initialize_match_service()
    if _service is None:
        raise HTTPException(status_code=500, detail=_service_init_error or "MatchService is not available.")
    return _service


def _get_identification_service() -> IdentificationService:
    _initialize_identification_service()
    if _ident_service is None:
        raise HTTPException(
            status_code=500,
            detail=_ident_service_init_error or "Identification service is not available.",
        )
    return _ident_service


def _get_browser_identification_service() -> IdentificationService:
    _initialize_browser_identification_service()
    if _browser_ident_service is None:
        raise HTTPException(
            status_code=500,
            detail=_browser_ident_service_init_error or "Browser identification service is not available.",
        )
    return _browser_ident_service


def _service_for_scope(store_scope: StoreScope = "operational") -> IdentificationService:
    if store_scope == "operational":
        return _get_identification_service()
    if store_scope == "browser":
        return _get_browser_identification_service()
    raise HTTPException(status_code=400, detail=f"Unsupported store_scope: {store_scope!r}.")


def _table_prefix_for_scope(store_scope: StoreScope, table_prefix: str | None = None) -> str:
    explicit_prefix = str(table_prefix or "").strip()
    if explicit_prefix:
        return explicit_prefix
    if store_scope == "operational":
        return ""
    if store_scope == "browser":
        return BROWSER_TABLE_PREFIX
    raise HTTPException(status_code=400, detail=f"Unsupported store_scope: {store_scope!r}.")


def _catalog_exception(exc: CatalogApiError) -> HTTPException:
    if isinstance(exc, CatalogInvalidRequestError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(
        exc,
        (
            CatalogArtifactError,
            CatalogBrowserAssetNotFoundError,
            CatalogDatasetNotFoundError,
            CatalogVerifyCaseNotFoundError,
        ),
    ):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _demo_exception(exc: DemoStoreError) -> HTTPException:
    if isinstance(exc, DemoInvalidSlotError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, (DemoCaseNotFoundError, DemoAssetResolutionError)):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _method_availability() -> dict[str, dict[str, object]]:
    if _service is not None:
        return _service.method_availability()
    if _service_init_error:
        return {
            definition.canonical_api_name: {
                "available": False,
                "error": _service_init_error,
            }
            for definition in load_api_method_registry().list_methods()
        }
    return {}


def _health_method_availability() -> dict[str, dict[str, object]]:
    availability = _method_availability()
    return {
        definition.canonical_api_name: availability.get(
            definition.canonical_api_name,
            {"available": None, "error": None},
        )
        for definition in load_api_method_registry().list_methods()
    }


def _methods_payload() -> dict[str, Any]:
    registry = load_api_method_registry()
    availability = _method_availability()
    entries = []
    for definition in registry.list_methods():
        state = availability.get(
            definition.canonical_api_name,
            {"available": True, "error": None},
        )
        entries.append(
            {
                "id": definition.canonical_api_name,
                "canonical_api_name": definition.canonical_api_name,
                "label": definition.ui_label,
                "benchmark_name": definition.benchmark_name,
                "aliases": list(definition.accepted_aliases),
                "family": definition.family,
                "status": definition.status,
                "embedding_dim": definition.embedding_dim,
                "thresholds": {"decision": definition.decision_threshold},
                "runtime_defaults": definition.runtime_defaults,
                "benchmark_defaults": definition.benchmark_defaults,
                "runtime_notes": list(definition.notes),
                "identification_roles": {
                    "retrieval_capable": definition.identification_role.retrieval_capable,
                    "rerank_capable": definition.identification_role.rerank_capable,
                    "notes": list(definition.identification_role.notes),
                },
                "availability": {
                    "available": bool(state.get("available", False)),
                    "error": state.get("error"),
                },
            }
        )
    return {"methods": entries}


def _browser_health_fields() -> dict[str, Any]:
    if _service_init_error is not None:
        return {
            "identify_browser_ok": False,
            "identify_browser_initialized": False,
            "identify_browser_status": "blocked",
            "identify_browser_error": f"MatchService failed: {_service_init_error}",
        }
    if _ident_service_init_error is not None:
        return {
            "identify_browser_ok": False,
            "identify_browser_initialized": False,
            "identify_browser_status": "error",
            "identify_browser_error": _ident_service_init_error,
        }
    if _browser_ident_service_init_error is not None:
        return {
            "identify_browser_ok": False,
            "identify_browser_initialized": False,
            "identify_browser_status": "error",
            "identify_browser_error": _browser_ident_service_init_error,
        }
    if _browser_ident_service is not None:
        return {
            "identify_browser_ok": True,
            "identify_browser_initialized": True,
            "identify_browser_status": "initialized",
            "identify_browser_error": None,
        }
    return {
        "identify_browser_ok": True,
        "identify_browser_initialized": False,
        "identify_browser_status": "lazy_not_initialized",
        "identify_browser_error": None,
    }


@router.get("/health")
def health() -> dict[str, Any]:
    if not _lazy_startup_enabled() or _service is not None or _service_init_error is not None:
        _initialize_match_service()
    if (
        not _lazy_startup_enabled()
        or _ident_service is not None
        or _ident_service_init_error is not None
        or _service is not None
        or _service_init_error is not None
    ):
        _initialize_identification_service()

    match_ok = _service is not None and _service_init_error is None
    if _lazy_startup_enabled() and _service is None and _service_init_error is None:
        match_ok = True
        match_status = "lazy_not_initialized"
    else:
        match_status = "ready" if match_ok else "error"

    identify_ok = _ident_service is not None and _ident_service_init_error is None
    if _service_init_error is not None:
        identify_status = "blocked"
        identify_error = f"MatchService failed: {_service_init_error}"
    elif _ident_service_init_error is not None:
        identify_status = "error"
        identify_error = _ident_service_init_error
    elif _lazy_startup_enabled() and _ident_service is None:
        identify_ok = True
        identify_status = "lazy_not_initialized"
        identify_error = None
    else:
        identify_status = "ready" if identify_ok else "lazy_not_initialized"
        identify_error = None

    return {
        "ok": match_ok,
        "status": match_status,
        "error": None if match_ok else _service_init_error,
        "identify_ok": identify_ok,
        "identify_status": identify_status,
        "identify_error": identify_error,
        **_browser_health_fields(),
        "methods": _health_method_availability(),
    }


@router.get("/methods")
def methods() -> dict[str, Any]:
    return _methods_payload()


@router.post("/match")
async def match(
    img_a: Annotated[UploadFile, File()],
    img_b: Annotated[UploadFile, File()],
    method: Annotated[str, Form()] = "classic",
    return_overlay: Annotated[bool, Form()] = False,
    capture_a: Annotated[str, Form()] = "plain",
    capture_b: Annotated[str, Form()] = "plain",
    threshold: Annotated[Optional[float], Form()] = None,
) -> MatchResponse:
    service = _get_match_service()
    path_a = await save_upload_to_temp(img_a, prefix="a", capture=capture_a)
    path_b = await save_upload_to_temp(img_b, prefix="b", capture=capture_b)
    try:
        return service.match(
            method=method,
            path_a=str(path_a),
            path_b=str(path_b),
            threshold=threshold,
            return_overlay=bool(return_overlay),
            capture_a=capture_a,
            capture_b=capture_b,
            filename_a=img_a.filename,
            filename_b=img_b.filename,
        )
    except (MethodRegistryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except api_service.MethodUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        path_a.unlink(missing_ok=True)
        path_b.unlink(missing_ok=True)


@router.get("/benchmark/runs")
def benchmark_runs():
    return load_benchmark_runs()


@router.get("/benchmark/summary")
def benchmark_summary(
    dataset: str | None = None,
    split: str | None = None,
    view_mode: str = "canonical",
):
    return load_benchmark_summary(dataset=dataset, split=split, view_mode=view_mode)


@router.get("/benchmark/comparison")
def benchmark_comparison(
    dataset: str | None = None,
    split: str | None = None,
    view_mode: str = "canonical",
    sort_mode: str = "best_accuracy",
):
    return load_comparison(dataset=dataset, split=split, view_mode=view_mode, sort_mode=sort_mode)


@router.get("/benchmark/best")
def benchmark_best(
    dataset: str | None = None,
    split: str | None = None,
    view_mode: str = "canonical",
):
    return load_best_methods(dataset=dataset, split=split, view_mode=view_mode)


@router.get("/benchmark/artifacts/{run}/{filename:path}")
def benchmark_artifact(run: str, filename: str) -> FileResponse:
    try:
        return FileResponse(resolve_benchmark_artifact(run, filename))
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/demo/cases")
def demo_cases() -> DemoCasesResponse:
    try:
        return load_demo_cases()
    except DemoStoreError as exc:
        raise _demo_exception(exc) from exc


@router.get("/demo/cases/{case_id}/{slot}")
def demo_case_asset(case_id: str, slot: str) -> FileResponse:
    try:
        return FileResponse(resolve_demo_case_path(case_id, slot))
    except DemoStoreError as exc:
        raise _demo_exception(exc) from exc


@router.get("/demo/cases/{case_id}/{slot}/{filename:path}")
def demo_case_asset_with_filename(case_id: str, slot: str, filename: str) -> FileResponse:
    del filename
    return demo_case_asset(case_id, slot)


@router.get("/catalog/datasets")
def catalog_datasets() -> CatalogDatasetsResponse:
    try:
        return load_catalog_datasets()
    except CatalogApiError as exc:
        raise _catalog_exception(exc) from exc


@router.get("/catalog/verify-cases")
def catalog_verify_cases(
    dataset: str | None = None,
    difficulty: str | None = None,
    tag: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> CatalogVerifyCasesResponse:
    try:
        return load_catalog_verify_cases(
            dataset=dataset,
            difficulty=difficulty,
            tag=tag,
            limit=limit,
            offset=offset,
        )
    except CatalogApiError as exc:
        raise _catalog_exception(exc) from exc


@router.get("/catalog/verify-cases/{case_id}")
def catalog_verify_case(case_id: str) -> CatalogVerifyCaseDetail:
    try:
        return load_catalog_verify_case_detail(case_id)
    except CatalogApiError as exc:
        raise _catalog_exception(exc) from exc


@router.get("/catalog/identify-gallery")
def catalog_identify_gallery(
    dataset: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> CatalogIdentifyGalleryResponse:
    try:
        return load_catalog_identify_gallery(dataset=dataset, limit=limit, offset=offset)
    except CatalogApiError as exc:
        raise _catalog_exception(exc) from exc


@router.get("/catalog/dataset-browser")
def catalog_dataset_browser(
    dataset: str,
    split: str | None = None,
    capture: str | None = None,
    modality: str | None = None,
    subject_id: str | None = None,
    finger: str | None = None,
    ui_eligible: bool | None = None,
    limit: int = 48,
    offset: int = 0,
    sort: str = "default",
) -> CatalogDatasetBrowserResponse:
    try:
        return load_catalog_dataset_browser(
            dataset=dataset,
            split=split,
            capture=capture,
            modality=modality,
            subject_id=subject_id,
            finger=finger,
            ui_eligible=ui_eligible,
            limit=limit,
            offset=offset,
            sort=sort,
        )
    except CatalogApiError as exc:
        raise _catalog_exception(exc) from exc


@router.get("/catalog/assets/{dataset}/{asset_id}/{variant}")
def catalog_asset(dataset: str, asset_id: str, variant: str) -> FileResponse:
    try:
        return FileResponse(resolve_catalog_browser_asset_path(dataset, asset_id, variant))
    except CatalogApiError as exc:
        raise _catalog_exception(exc) from exc


@router.get("/identify/stats")
def identify_stats() -> IdentificationStatsResponse:
    service = _get_identification_service()
    browser_seeded_count = 0
    if _browser_ident_service is not None:
        browser_seeded_count = get_identify_browser_seeded_count(_browser_ident_service)

    return IdentificationStatsResponse(
        **service.stats(),
        demo_seeded_count=get_identify_demo_seeded_count(service),
        browser_seeded_count=browser_seeded_count,
    )


def _collect_identification_admin_state(
    *,
    store_scope: StoreScope = "operational",
    table_prefix: str | None = None,
) -> dict[str, object]:
    return SecureSplitFingerprintStore.inspect_runtime_state(
        database_url=None,
        identity_database_url=None,
        table_prefix=_table_prefix_for_scope(store_scope, table_prefix),
    )


def _collect_identification_admin_reconciliation_report(
    *,
    store_scope: StoreScope = "operational",
    table_prefix: str | None = None,
) -> dict[str, object]:
    return SecureSplitFingerprintStore.reconcile_runtime_state(
        database_url=None,
        identity_database_url=None,
        table_prefix=_table_prefix_for_scope(store_scope, table_prefix),
    )


@router.get("/identify/admin/layout")
def identify_admin_layout(
    store_scope: StoreScope = "operational",
    table_prefix: str | None = None,
) -> IdentificationAdminInspectionResponse:
    try:
        return IdentificationAdminInspectionResponse(
            **_collect_identification_admin_state(
                store_scope=store_scope,
                table_prefix=table_prefix,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/identify/admin/reconcile")
def identify_admin_reconcile(
    store_scope: StoreScope = "operational",
    table_prefix: str | None = None,
) -> IdentificationAdminReconciliationResponse:
    try:
        return IdentificationAdminReconciliationResponse(
            **_collect_identification_admin_reconciliation_report(
                store_scope=store_scope,
                table_prefix=table_prefix,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/identify/enroll")
async def identify_enroll(
    img: Annotated[UploadFile, File()],
    full_name: Annotated[str, Form()],
    national_id: Annotated[str, Form()],
    capture: Annotated[str, Form()] = "plain",
    vector_methods: Annotated[str, Form()] = "dl,vit",
    replace_existing: Annotated[bool, Form()] = False,
) -> EnrollFingerprintResponse:
    service = _get_identification_service()
    path = await save_upload_to_temp(img, prefix="enroll", capture=capture)
    try:
        methods = [item.strip() for item in vector_methods.split(",") if item.strip()]
        receipt = service.enroll_from_path(
            path=str(path),
            full_name=full_name,
            national_id=national_id,
            capture=capture,
            vector_methods=methods,
            replace_existing=replace_existing,
        )
        return EnrollFingerprintResponse(
            **asdict(receipt),
            storage_layout=service.store.dump_layout(),
        )
    except (ValueError, FileNotFoundError, MethodRegistryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@router.post("/identify/search")
async def identify_search(
    img: Annotated[UploadFile, File()],
    capture: Annotated[str, Form()] = "plain",
    retrieval_method: Annotated[str, Form()] = "dl",
    rerank_method: Annotated[MatchMethod, Form()] = MatchMethod.sift,
    shortlist_size: Annotated[int, Form()] = 25,
    threshold: Annotated[Optional[float], Form()] = None,
    name_pattern: Annotated[Optional[str], Form()] = None,
    national_id_pattern: Annotated[Optional[str], Form()] = None,
    created_from: Annotated[Optional[str], Form()] = None,
    created_to: Annotated[Optional[str], Form()] = None,
    store_scope: Annotated[StoreScope, Form()] = "operational",
) -> IdentifyResponse:
    service = _service_for_scope(store_scope)
    path = await save_upload_to_temp(img, prefix="identify", capture=capture)
    try:
        result = service.identify_from_path(
            path=str(path),
            capture=capture,
            retrieval_method=retrieval_method,
            rerank_method=rerank_method,
            shortlist_size=shortlist_size,
            threshold=threshold,
            hints=IdentifyHints(
                name_pattern=name_pattern,
                national_id_pattern=national_id_pattern,
                created_from=created_from,
                created_to=created_to,
            ),
        )
        return IdentifyResponse(**asdict(result))
    except (ValueError, FileNotFoundError, MethodRegistryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@router.delete("/identify/person/{random_id}")
def identify_delete_person(random_id: str) -> DeleteIdentityResponse:
    service = _get_identification_service()
    return DeleteIdentityResponse(
        random_id=random_id,
        removed=bool(service.store.purge(random_id)),
        storage_layout=service.store.dump_layout(),
    )


@router.post("/identify/demo/seed")
def identify_demo_seed(dataset: str | None = None) -> IdentifyDemoSeedResponse:
    service = _get_identification_service()
    try:
        return IdentifyDemoSeedResponse(**seed_identify_demo_store(service, dataset=dataset))
    except (IdentifyDemoStoreError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/identify/demo/reset")
def identify_demo_reset(dataset: str | None = None) -> IdentifyDemoResetResponse:
    service = _get_identification_service()
    try:
        return IdentifyDemoResetResponse(**reset_identify_demo_store(service, dataset=dataset))
    except (IdentifyDemoStoreError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/identify/browser/seed-selection")
def identify_browser_seed_selection(
    request: IdentifyBrowserSeedSelectionRequest,
) -> IdentifyBrowserSeedSelectionResponse:
    service = _get_browser_identification_service()
    try:
        return IdentifyBrowserSeedSelectionResponse(
            **seed_identify_browser_store(
                service,
                dataset=request.dataset,
                selected_identity_ids=request.selected_identity_ids,
                overwrite=request.overwrite,
                metadata=request.metadata,
            )
        )
    except (IdentifyDemoStoreError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/identify/browser/reset")
def identify_browser_reset() -> IdentifyBrowserResetResponse:
    service = _get_browser_identification_service()
    try:
        return IdentifyBrowserResetResponse(**reset_identify_browser_store(service))
    except (IdentifyDemoStoreError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.include_router(router)
app.include_router(router, prefix="/api", include_in_schema=False)
