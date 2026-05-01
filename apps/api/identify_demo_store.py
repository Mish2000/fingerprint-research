from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from apps.api.catalog_store import CatalogApiError, load_catalog_identify_demo_identity_records, load_catalog_identify_seed_records
from apps.api.identification_service import IdentificationService

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDENTIFY_DEMO_STATE_PATH = ROOT / "artifacts" / "runtime" / "identify_demo_store_state.json"
DEFAULT_IDENTIFY_BROWSER_STATE_PATH = ROOT / "artifacts" / "runtime" / "identify_browser_store_state.json"
_DEMO_ID_PREFIX_RE = re.compile(r"[^a-z0-9_]+")


class IdentifyDemoStoreError(RuntimeError):
    pass


class IdentifyDemoStateError(IdentifyDemoStoreError):
    pass


class IdentifyDemoCatalogError(IdentifyDemoStoreError):
    pass


def _demo_state_path() -> Path:
    override = os.getenv("FPBENCH_IDENTIFY_DEMO_STATE_PATH")
    return Path(override).resolve() if override else DEFAULT_IDENTIFY_DEMO_STATE_PATH.resolve()


def _browser_state_path() -> Path:
    override = os.getenv("FPBENCH_IDENTIFY_BROWSER_STATE_PATH")
    return Path(override).resolve() if override else DEFAULT_IDENTIFY_BROWSER_STATE_PATH.resolve()


def _normalize_demo_id(value: str) -> str:
    return _DEMO_ID_PREFIX_RE.sub("_", str(value).strip().lower()).strip("_")


def _demo_random_id(dataset: str, identity_id: str) -> str:
    dataset_key = _normalize_demo_id(dataset)
    identity_key = _normalize_demo_id(identity_id)
    return f"demo_identify_{dataset_key}_{identity_key}"


def _browser_random_id(dataset: str, identity_id: str) -> str:
    dataset_key = _normalize_demo_id(dataset)
    identity_key = _normalize_demo_id(identity_id)
    return f"browser_identify_{dataset_key}_{identity_key}"


def _demo_national_id(dataset: str, identity_id: str) -> str:
    digest = hashlib.sha256(f"{dataset}:{identity_id}".encode("utf-8")).hexdigest()
    numeric = int(digest[:16], 16) % 10**16
    return f"99{numeric:016d}"


def _demo_created_at(index: int) -> str:
    base = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
    return (base + timedelta(minutes=index)).isoformat()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"seeded_random_ids": [], "seeded_identity_ids": []}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IdentifyDemoStateError(f"Identify demo store state is not valid JSON: {path}") from exc
    except OSError as exc:
        raise IdentifyDemoStateError(f"Could not read identify demo store state: {path}") from exc

    if not isinstance(payload, dict):
        raise IdentifyDemoStateError("Identify demo store state must be a JSON object.")

    seeded_random_ids = payload.get("seeded_random_ids", [])
    seeded_identity_ids = payload.get("seeded_identity_ids", [])
    if not isinstance(seeded_random_ids, list) or not all(isinstance(item, str) for item in seeded_random_ids):
        raise IdentifyDemoStateError("Identify demo store state seeded_random_ids must be a list of strings.")
    if not isinstance(seeded_identity_ids, list) or not all(isinstance(item, str) for item in seeded_identity_ids):
        raise IdentifyDemoStateError("Identify demo store state seeded_identity_ids must be a list of strings.")

    return {
        "seeded_random_ids": seeded_random_ids,
        "seeded_identity_ids": seeded_identity_ids,
    }


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_state(path: Path) -> None:
    path.unlink(missing_ok=True)


def _gallery_demo_random_ids(*, dataset: Optional[str] = None) -> list[str]:
    try:
        records = load_catalog_identify_demo_identity_records(dataset=dataset)
    except CatalogApiError as exc:
        raise IdentifyDemoCatalogError(str(exc)) from exc
    return [_demo_random_id(record.public_item.dataset, record.public_item.id) for record in records]


def get_identify_demo_seeded_count(
    service: IdentificationService,
    *,
    dataset: Optional[str] = None,
) -> int:
    state = _load_state(_demo_state_path())
    known_random_ids = {
        random_id
        for random_id in [*state["seeded_random_ids"], *_gallery_demo_random_ids(dataset=dataset)]
        if random_id.startswith("demo_identify_")
    }
    count = 0
    for random_id in known_random_ids:
        if service.store.get_person(random_id) is not None:
            count += 1
    return count


def seed_identify_demo_store(
    service: IdentificationService,
    *,
    dataset: Optional[str] = None,
) -> dict[str, Any]:
    try:
        records = load_catalog_identify_demo_identity_records(dataset=dataset)
    except CatalogApiError as exc:
        raise IdentifyDemoCatalogError(str(exc)) from exc

    if not records:
        demo_seeded_count = get_identify_demo_seeded_count(service, dataset=dataset)
        return {
            "seeded_count": 0,
            "updated_count": 0,
            "skipped_count": 0,
            "total_enrolled": service.store.total_people(),
            "demo_seeded_count": demo_seeded_count,
            "storage_layout": service.store.dump_layout(),
            "notice": "No demo identities are currently available for seeding.",
        }

    seeded_count = 0
    updated_count = 0
    seeded_random_ids: list[str] = []
    seeded_identity_ids: list[str] = []

    for index, record in enumerate(records):
        random_id = _demo_random_id(record.public_item.dataset, record.public_item.id)
        national_id = _demo_national_id(record.public_item.dataset, record.public_item.id)
        existed = service.store.get_person(random_id) is not None
        if existed:
            service.store.purge(random_id)
            updated_count += 1
        else:
            seeded_count += 1

        service.enroll_from_path(
            path=str(record.enrollment_asset_path),
            full_name=record.public_item.display_label,
            national_id=national_id,
            capture=record.enrollment_capture,
            vector_methods=("dl", "vit"),
            replace_existing=False,
            random_id=random_id,
            created_at=_demo_created_at(index),
        )
        seeded_random_ids.append(random_id)
        seeded_identity_ids.append(record.public_item.id)

    _write_state(
        _demo_state_path(),
        {
            "seeded_random_ids": seeded_random_ids,
            "seeded_identity_ids": seeded_identity_ids,
        }
    )
    demo_seeded_count = get_identify_demo_seeded_count(service, dataset=dataset)
    return {
        "seeded_count": seeded_count,
        "updated_count": updated_count,
        "skipped_count": 0,
        "total_enrolled": service.store.total_people(),
        "demo_seeded_count": demo_seeded_count,
        "storage_layout": service.store.dump_layout(),
        "notice": f"Demo store now contains {demo_seeded_count} seeded identity{'ies' if demo_seeded_count != 1 else ''}.",
    }


def reset_identify_demo_store(
    service: IdentificationService,
    *,
    dataset: Optional[str] = None,
) -> dict[str, Any]:
    state = _load_state(_demo_state_path())
    candidate_random_ids = {
        random_id
        for random_id in [*state["seeded_random_ids"], *_gallery_demo_random_ids(dataset=dataset)]
        if random_id.startswith("demo_identify_")
    }

    removed_count = 0
    for random_id in candidate_random_ids:
        if service.store.purge(random_id):
            removed_count += 1

    _clear_state(_demo_state_path())
    demo_seeded_count = get_identify_demo_seeded_count(service, dataset=dataset)
    return {
        "removed_count": removed_count,
        "total_enrolled": service.store.total_people(),
        "demo_seeded_count": demo_seeded_count,
        "storage_layout": service.store.dump_layout(),
        "notice": "Demo store reset completed. Only demo-seeded identities were targeted.",
    }


def get_identify_browser_seeded_count(
    service: IdentificationService,
    *,
    dataset: Optional[str] = None,
) -> int:
    state = _load_state(_browser_state_path())
    known_random_ids = {
        random_id
        for random_id in state["seeded_random_ids"]
        if random_id.startswith("browser_identify_")
    }
    count = 0
    for random_id in known_random_ids:
        if dataset and not random_id.startswith(f"browser_identify_{_normalize_demo_id(dataset)}_"):
            continue
        if service.store.get_person(random_id) is not None:
            count += 1
    return count


def seed_identify_browser_store(
    service: IdentificationService,
    *,
    dataset: str,
    selected_identity_ids: list[str],
    overwrite: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    dataset_value = str(dataset).strip()
    normalized_selected_ids = [
        str(identity_id).strip()
        for identity_id in selected_identity_ids
        if str(identity_id).strip()
    ]
    if not dataset_value:
        raise ValueError("dataset is required for browser seeding.")
    if not normalized_selected_ids:
        raise ValueError("selected_identity_ids must contain at least one identity.")

    try:
        records = load_catalog_identify_seed_records(
            dataset=dataset_value,
            selected_identity_ids=normalized_selected_ids,
        )
    except CatalogApiError as exc:
        raise IdentifyDemoCatalogError(str(exc)) from exc

    selected_lookup = set(normalized_selected_ids)
    warnings: list[str] = []
    errors: list[str] = []

    if overwrite:
        reset_identify_browser_store(service)

    if not records:
        errors.append(f"No browser-seedable identities were resolved for dataset {dataset_value!r}.")
        browser_seeded_count = get_identify_browser_seeded_count(service, dataset=dataset_value)
        return {
            "dataset": dataset_value,
            "selected_count": len(normalized_selected_ids),
            "seeded_count": 0,
            "updated_count": 0,
            "skipped_count": len(normalized_selected_ids),
            "total_enrolled": service.store.total_people(),
            "browser_seeded_count": browser_seeded_count,
            "store_ready": browser_seeded_count > 0,
            "seeded_identity_ids": [],
            "storage_layout": service.store.dump_layout(),
            "warnings": warnings,
            "errors": errors,
            "notice": "Browser store was not updated because none of the requested identities were seedable.",
        }

    resolved_lookup = {record.public_item.identity_id: record for record in records}
    unresolved_ids = [identity_id for identity_id in normalized_selected_ids if identity_id not in resolved_lookup]
    if unresolved_ids:
        warnings.extend(
            f"Identity {identity_id!r} could not be seeded from the catalog-backed browser gallery."
            for identity_id in unresolved_ids
        )

    seeded_count = 0
    updated_count = 0
    seeded_random_ids: list[str] = []
    seeded_identity_ids: list[str] = []

    for index, identity_id in enumerate(normalized_selected_ids):
        record = resolved_lookup.get(identity_id)
        if record is None:
            continue

        random_id = _browser_random_id(record.public_item.dataset, record.public_item.identity_id)
        national_id = _demo_national_id(record.public_item.dataset, record.public_item.identity_id)
        existed = service.store.get_person(random_id) is not None
        if existed:
            service.store.purge(random_id)
            updated_count += 1
        else:
            seeded_count += 1

        service.enroll_from_path(
            path=str(record.enrollment_asset_path),
            full_name=record.public_item.display_name,
            national_id=national_id,
            capture=record.enrollment_capture,
            vector_methods=("dl", "vit"),
            replace_existing=False,
            random_id=random_id,
            created_at=_demo_created_at(index),
        )
        seeded_random_ids.append(random_id)
        seeded_identity_ids.append(record.public_item.identity_id)

    _write_state(
        _browser_state_path(),
        {
            "dataset": dataset_value,
            "seeded_random_ids": seeded_random_ids,
            "seeded_identity_ids": seeded_identity_ids,
            "metadata": metadata or {},
        },
    )

    skipped_count = max(len(selected_lookup) - len(seeded_identity_ids), 0)
    browser_seeded_count = get_identify_browser_seeded_count(service, dataset=dataset_value)
    return {
        "dataset": dataset_value,
        "selected_count": len(normalized_selected_ids),
        "seeded_count": seeded_count,
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "total_enrolled": service.store.total_people(),
        "browser_seeded_count": browser_seeded_count,
        "store_ready": browser_seeded_count > 0 and not errors,
        "seeded_identity_ids": seeded_identity_ids,
        "storage_layout": service.store.dump_layout(),
        "warnings": warnings,
        "errors": errors,
        "notice": (
            f"Browser store prepared with {browser_seeded_count} selected identity"
            f"{'ies' if browser_seeded_count != 1 else 'y'}."
        ),
    }


def reset_identify_browser_store(
    service: IdentificationService,
) -> dict[str, Any]:
    state = _load_state(_browser_state_path())
    candidate_random_ids = {
        random_id
        for random_id in state["seeded_random_ids"]
        if random_id.startswith("browser_identify_")
    }

    removed_count = 0
    for random_id in candidate_random_ids:
        if service.store.purge(random_id):
            removed_count += 1

    _clear_state(_browser_state_path())
    browser_seeded_count = get_identify_browser_seeded_count(service)
    return {
        "removed_count": removed_count,
        "total_enrolled": service.store.total_people(),
        "browser_seeded_count": browser_seeded_count,
        "storage_layout": service.store.dump_layout(),
        "notice": "Browser-seeded store reset completed. Operational enrollments were not touched.",
    }
