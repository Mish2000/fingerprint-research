from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import apps.api.demo_store as demo_store
from apps.api.benchmark_meta import normalize_benchmark_context
from apps.api.demo_store import CatalogAssetModel, CatalogSourceDatasetModel, CatalogVerifyCaseModel
from apps.api.method_registry import load_api_method_registry
from apps.api.public_health import build_catalog_build_health_summary, build_dataset_demo_health_summary
from apps.api.schemas import (
    AssetDimensions,
    CatalogBuildHealthSummary,
    CatalogBrowserItem,
    CatalogDatasetBrowserResponse,
    CatalogDatasetDemoHealth,
    CatalogDatasetSummary,
    CatalogDatasetsResponse,
    CatalogIdentityItem,
    CatalogIdentifyDemoIdentity,
    CatalogIdentifyExemplar,
    CatalogIdentifyGalleryResponse,
    CatalogIdentifyProbeCase,
    CatalogVerifyCaseDetail,
    CatalogVerifyCaseItem,
    CatalogVerifyCasesResponse,
    MatchMethod,
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLES_ROOT = ROOT / "data" / "samples"
CATALOG_PATH = SAMPLES_ROOT / "catalog.json"
ASSETS_ROOT = SAMPLES_ROOT / "assets"
PROCESSED_ROOT = ROOT / "data" / "processed"
UI_ASSETS_REGISTRY_PATH = PROCESSED_ROOT / "ui_assets_registry.json"

DEFAULT_PAGE_LIMIT = 20
DEFAULT_BROWSER_LIMIT = 48
MAX_PAGE_LIMIT = 200
ALLOWED_BROWSER_SORTS = {"default", "split_subject_asset"}

logger = logging.getLogger(__name__)


class CatalogApiError(RuntimeError):
    pass


class CatalogArtifactError(CatalogApiError):
    pass


class CatalogInvalidRequestError(CatalogApiError):
    pass


class CatalogDatasetNotFoundError(CatalogApiError):
    pass


class CatalogVerifyCaseNotFoundError(CatalogApiError):
    pass


class CatalogBrowserAssetNotFoundError(CatalogApiError):
    pass


class CatalogIdentifyGalleryModel(BaseModel):
    identities: list[Any] = Field(default_factory=list)
    demo_scenarios: list[Any] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class CatalogIdentityRecordModel(BaseModel):
    identity_id: str
    dataset: str
    display_name: str
    subject_id: Any
    gallery_role: str
    enrollment_candidates: list[str] = Field(default_factory=list)
    probe_candidates: list[str] = Field(default_factory=list)
    recommended_enrollment_asset_id: Optional[str] = None
    recommended_probe_asset_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    is_demo_safe: bool
    exemplars: list[CatalogAssetModel] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class CatalogIdentifyScenarioRecordModel(BaseModel):
    scenario_id: str
    scenario_type: str
    dataset: str
    title: str
    description: str
    enrollment_identity_ids: list[str] = Field(default_factory=list)
    probe_asset: CatalogAssetModel
    expected_identity_id: Optional[str] = None
    difficulty: str
    recommended_method: str
    tags: list[str] = Field(default_factory=list)
    is_demo_safe: bool

    model_config = ConfigDict(extra="allow")


class CatalogPayloadModel(BaseModel):
    source_datasets: list[Any] = Field(default_factory=list)
    verify_cases: list[Any] = Field(default_factory=list)
    identify_gallery: CatalogIdentifyGalleryModel = Field(default_factory=CatalogIdentifyGalleryModel)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class UiAssetsRegistryEntryModel(BaseModel):
    dataset: str
    index_path: str
    validation_report_path: str
    item_count: int
    validation_status: str
    selection_policy: str
    summary: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class UiAssetsRegistryPayloadModel(BaseModel):
    generated_at: Optional[str] = None
    generator_version: Optional[str] = None
    datasets: list[UiAssetsRegistryEntryModel] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class UiAssetDimensionsModel(BaseModel):
    width: int
    height: int


class UiAssetsIndexItemModel(BaseModel):
    asset_id: str
    dataset: str
    split: str
    source_path: str
    thumbnail_path: str
    preview_path: str
    availability_status: str
    subject_id: Optional[Any] = None
    finger: Optional[str] = None
    capture: Optional[str] = None
    modality: Optional[str] = None
    traceability: dict[str, Any] = Field(default_factory=dict)
    ui_eligible: bool
    selection_reason: str
    selection_policy: str
    original_dimensions: UiAssetDimensionsModel
    thumbnail_dimensions: UiAssetDimensionsModel
    preview_dimensions: UiAssetDimensionsModel

    model_config = ConfigDict(extra="allow")


class UiAssetsIndexPayloadModel(BaseModel):
    dataset: str
    dataset_label: str
    generated_at: Optional[str] = None
    generator_version: Optional[str] = None
    selection_policy: str
    items: list[Any] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    validation_status: str

    model_config = ConfigDict(extra="allow")


class UiAssetsValidationReportModel(BaseModel):
    dataset: str
    dataset_label: Optional[str] = None
    generated_at: Optional[str] = None
    generator_version: Optional[str] = None
    selection_policy: Optional[str] = None
    source_records_checked: Optional[int] = None
    generated_items: Optional[int] = None
    excluded_records: int = 0
    missing_source_files: int = 0
    unreadable_source_files: int = 0
    missing_critical_metadata: int = 0
    duplicates_skipped: int = 0
    validation_status: str

    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class CatalogState:
    dataset_labels: Dict[str, str] = field(default_factory=dict)
    catalog_datasets: set[str] = field(default_factory=set)
    verify_items: list[CatalogVerifyCaseItem] = field(default_factory=list)
    verify_details: Dict[str, CatalogVerifyCaseDetail] = field(default_factory=dict)
    catalog_build_health: Optional[CatalogBuildHealthSummary] = None
    dataset_demo_health: Dict[str, CatalogDatasetDemoHealth] = field(default_factory=dict)
    identity_items: list[CatalogIdentityItem] = field(default_factory=list)
    identity_seed_records: Dict[str, "IdentifySeedIdentityRecord"] = field(default_factory=dict)
    demo_identity_items: list[CatalogIdentifyDemoIdentity] = field(default_factory=list)
    probe_case_items: list[CatalogIdentifyProbeCase] = field(default_factory=list)
    demo_identity_records: Dict[str, "IdentifyDemoIdentityRecord"] = field(default_factory=dict)
    probe_case_records: Dict[str, "IdentifyProbeCaseRecord"] = field(default_factory=dict)


@dataclass(frozen=True)
class IdentifyDemoIdentityRecord:
    public_item: CatalogIdentifyDemoIdentity
    enrollment_asset_path: Path
    enrollment_capture: str


@dataclass(frozen=True)
class IdentifySeedIdentityRecord:
    public_item: CatalogIdentityItem
    enrollment_asset_path: Path
    enrollment_capture: str


@dataclass(frozen=True)
class IdentifyProbeCaseRecord:
    public_item: CatalogIdentifyProbeCase
    probe_asset_path: Path


@dataclass(frozen=True)
class BrowserAssetRecord:
    public_item: CatalogBrowserItem
    asset_paths: Dict[str, Path]
    index_position: int
    split_subject_sort_key: tuple[Any, ...]


@dataclass(frozen=True)
class BrowserDatasetState:
    dataset: str
    dataset_label: str
    selection_policy: str
    validation_status: str
    generated_at: Optional[str]
    generator_version: Optional[str]
    warning_count: int
    summary: Dict[str, Any] = field(default_factory=dict)
    items: list[BrowserAssetRecord] = field(default_factory=list)
    items_by_id: Dict[str, BrowserAssetRecord] = field(default_factory=dict)
    items_by_source_key: Dict[str, BrowserAssetRecord] = field(default_factory=dict)
    excluded_asset_reasons: Dict[str, str] = field(default_factory=dict)


def _repo_root() -> Path:
    override = os.getenv("FPBENCH_DEMO_ROOT")
    return Path(override).resolve() if override else ROOT.resolve()


def _catalog_path() -> Path:
    override = os.getenv("FPBENCH_DEMO_CATALOG_PATH")
    return Path(override).resolve() if override else CATALOG_PATH.resolve()


def _assets_root() -> Path:
    override = os.getenv("FPBENCH_DEMO_ASSETS_ROOT")
    return Path(override).resolve() if override else ASSETS_ROOT.resolve()


def _processed_root() -> Path:
    override = os.getenv("FPBENCH_UI_ASSETS_ROOT")
    return Path(override).resolve() if override else PROCESSED_ROOT.resolve()


def _ui_assets_registry_path() -> Path:
    override = os.getenv("FPBENCH_UI_ASSETS_REGISTRY_PATH")
    return Path(override).resolve() if override else UI_ASSETS_REGISTRY_PATH.resolve()


def _path_is_under(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _looks_absolute(path_str: str) -> bool:
    return PurePosixPath(path_str).is_absolute() or PureWindowsPath(path_str).is_absolute()


def _normalize_repo_relative_path(path_str: str, *, allowed_root: Path) -> Path:
    if not path_str:
        raise CatalogArtifactError("Artifact path is missing.")

    normalized = str(path_str).replace("\\", "/").strip()
    if not normalized:
        raise CatalogArtifactError("Artifact path is empty.")
    if _looks_absolute(normalized):
        raise CatalogArtifactError("Absolute artifact paths are not allowed.")

    logical = PurePosixPath(normalized)
    if any(part == ".." for part in logical.parts):
        raise CatalogArtifactError("Artifact path traversal is not allowed.")

    if normalized.startswith("data/"):
        candidate = _repo_root().joinpath(*logical.parts)
    else:
        candidate = allowed_root.joinpath(*logical.parts)

    resolved = candidate.resolve()
    if not _path_is_under(allowed_root, resolved):
        raise CatalogArtifactError("Resolved artifact path falls outside the allowed root.")
    return resolved


def _resolve_catalog_asset_path(asset: CatalogAssetModel) -> Path:
    if str(asset.availability_status) != "available":
        raise CatalogArtifactError(f"Catalog asset {asset.asset_id} is not available.")

    logical_path = asset.relative_path or asset.path
    path = _normalize_repo_relative_path(logical_path, allowed_root=_assets_root())
    if not path.is_file():
        raise CatalogArtifactError(f"Catalog asset file is missing for {asset.asset_id}.")
    return path


def _resolve_browser_index_path(path_str: str) -> Path:
    path = _normalize_repo_relative_path(path_str, allowed_root=_processed_root())
    if not path.is_file():
        raise CatalogArtifactError(f"UI assets artifact is missing: {path}")
    return path


def _resolve_browser_asset_path(dataset: str, path_str: str) -> Path:
    dataset_ui_root = (_processed_root() / dataset / "ui_assets").resolve()
    path = _normalize_repo_relative_path(path_str, allowed_root=dataset_ui_root)
    if not path.is_file():
        raise CatalogBrowserAssetNotFoundError(f"Browser asset file is missing for dataset {dataset!r}.")
    if not _path_is_under(dataset_ui_root, path):
        raise CatalogBrowserAssetNotFoundError("Resolved browser asset path violates ui_assets root policy.")
    return path


def _safe_browser_asset_url(dataset: str, asset_id: str, variant: str) -> str:
    return f"/api/catalog/assets/{dataset}/{asset_id}/{variant}"


def _normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_subject_id(value: Any) -> str:
    text = _normalize_optional_text(value)
    return text or "unknown"


def _case_insensitive_equals(left: Any, right: Any) -> bool:
    return (_normalize_optional_text(left) or "").lower() == (_normalize_optional_text(right) or "").lower()


def _normalize_limit_offset(limit: int, offset: int, *, default_limit: int) -> tuple[int, int]:
    if limit < 0:
        raise CatalogInvalidRequestError("limit must be greater than or equal to 0.")
    if offset < 0:
        raise CatalogInvalidRequestError("offset must be greater than or equal to 0.")
    effective_limit = default_limit if limit == 0 else min(limit, MAX_PAGE_LIMIT)
    return effective_limit, offset


def _paginate(items: list[Any], *, limit: int, offset: int) -> tuple[list[Any], bool]:
    window = items[offset : offset + limit]
    has_more = offset + limit < len(items)
    return window, has_more


def _asset_dimensions(model: UiAssetDimensionsModel) -> AssetDimensions:
    return AssetDimensions(width=model.width, height=model.height)


def _build_verify_traceability(case: CatalogVerifyCaseModel) -> dict[str, Any]:
    image_a_traceability = (case.image_a.model_extra or {}).get("traceability") or {}
    image_b_traceability = (case.image_b.model_extra or {}).get("traceability") or {}
    return {
        "asset_a_id": case.image_a.asset_id,
        "asset_b_id": case.image_b.asset_id,
        "dataset": case.dataset,
        "split": case.split,
        "image_a_source_dataset": image_a_traceability.get("source_dataset"),
        "image_b_source_dataset": image_b_traceability.get("source_dataset"),
        "image_a_materialized_kind": image_a_traceability.get("materialized_asset_kind"),
        "image_b_materialized_kind": image_b_traceability.get("materialized_asset_kind"),
    }


def _build_verify_notes(case: CatalogVerifyCaseModel) -> dict[str, Any]:
    extras = case.model_extra or {}
    allowed_keys = (
        "story_label",
        "difficulty_reason",
        "curation_rationale",
        "additional_notes",
        "scenario_type",
    )
    return {key: extras[key] for key in allowed_keys if key in extras}


def _normalize_lookup_path_key(value: Any) -> Optional[str]:
    text = _normalize_optional_text(value)
    if not text:
        return None

    normalized = text.replace("\\", "/").strip()
    if not normalized:
        return None

    if _looks_absolute(normalized):
        try:
            candidate = Path(normalized).resolve()
            normalized = candidate.relative_to(_repo_root()).as_posix()
        except Exception:
            pass

    return str(PurePosixPath(normalized)).casefold()


def _browser_lookup_keys_for_ui_item(item: UiAssetsIndexItemModel) -> set[str]:
    keys = {
        key
        for key in {
            _normalize_lookup_path_key(item.source_path),
            _normalize_lookup_path_key((item.traceability or {}).get("source_path_relative")),
            _normalize_lookup_path_key((item.traceability or {}).get("source_path_original")),
        }
        if key
    }
    return keys


def _catalog_asset_lookup_keys(asset: CatalogAssetModel) -> set[str]:
    extras = asset.model_extra or {}
    return {
        key
        for key in {
            _normalize_lookup_path_key(extras.get("source_relative_path")),
            _normalize_lookup_path_key(extras.get("source_path")),
        }
        if key
    }


def _resolve_browser_record_for_catalog_asset(asset: CatalogAssetModel) -> Optional[BrowserAssetRecord]:
    dataset = _normalize_optional_text((asset.model_extra or {}).get("dataset"))
    if not dataset:
        return None

    try:
        state = _get_browser_dataset_state(dataset)
    except CatalogApiError:
        return None

    for lookup_key in _catalog_asset_lookup_keys(asset):
        record = state.items_by_source_key.get(lookup_key)
        if record is not None:
            return record
    return None


def _resolve_identification_methods(recommended_method: str | None) -> tuple[str, MatchMethod]:
    raw_method = str(recommended_method or "").strip().lower()
    if raw_method:
        registry = load_api_method_registry()
        try:
            resolved = registry.resolve(raw_method)
        except Exception:
            resolved = None
        if resolved is not None:
            retrieval_method = (
                resolved.canonical_api_name
                if resolved.definition.identification_role.retrieval_capable
                else "dl"
            )
            return retrieval_method, MatchMethod(resolved.canonical_api_name)
    return "dl", MatchMethod.sift


def _expected_outcome_for_scenario(scenario: CatalogIdentifyScenarioRecordModel) -> str:
    scenario_type = str(scenario.scenario_type).strip().lower()
    if scenario.expected_identity_id and "no_match" not in scenario_type:
        return "match"
    return "no_match"


def _probe_case_sort_key(item: CatalogIdentifyProbeCase) -> tuple[Any, ...]:
    scenario_type = str(item.scenario_type or "").strip().lower()
    difficulty = str(item.difficulty or "").strip().lower()
    if scenario_type == "positive_identification" and difficulty == "easy":
        order = 0
    elif scenario_type == "positive_identification":
        order = 1
    elif scenario_type == "difficult_identification":
        order = 2
    elif scenario_type == "no_match":
        order = 3
    elif "modality_special" in {tag.lower() for tag in item.tags}:
        order = 4
    else:
        order = 5
    return (order, item.dataset_label.lower(), difficulty, item.title.lower(), item.id)


def _catalog_asset_id_from_api_url(url: str) -> str | None:
    parts = [part for part in str(url).split("/") if part]
    if len(parts) < 2:
        return None
    try:
        assets_index = parts.index("assets")
    except ValueError:
        return None
    if len(parts) <= assets_index + 2:
        return None
    return parts[assets_index + 2]


def _build_identify_exemplar(
    asset_id: str,
    exemplars_by_id: Mapping[str, CatalogAssetModel],
    *,
    fallback_usage: str,
) -> CatalogIdentifyExemplar:
    exemplar = exemplars_by_id.get(asset_id)
    recommended_usage = fallback_usage
    availability_status: Optional[str] = None
    capture: Optional[str] = None
    finger: Optional[str] = None
    has_servable_asset = False

    if exemplar is not None:
        capture = exemplar.capture
        finger = _normalize_optional_text((exemplar.model_extra or {}).get("finger"))
        availability_status = exemplar.availability_status
        recommended_usage = (exemplar.model_extra or {}).get("recommended_usage") or fallback_usage
        try:
            _resolve_catalog_asset_path(exemplar)
        except CatalogArtifactError:
            has_servable_asset = False
        else:
            has_servable_asset = True

    return CatalogIdentifyExemplar(
        asset_id=asset_id,
        capture=capture,
        finger=finger,
        recommended_usage=recommended_usage,
        asset_reference=asset_id,
        has_servable_asset=has_servable_asset,
        availability_status=availability_status,
    )


def _preferred_identity_asset_ids(identity: CatalogIdentityRecordModel) -> list[str]:
    return [
        asset_id
        for asset_id in [
            identity.recommended_enrollment_asset_id,
            *identity.enrollment_candidates,
            identity.recommended_probe_asset_id,
            *identity.probe_candidates,
        ]
        if asset_id
    ]


def _build_identity_item(
    identity: CatalogIdentityRecordModel,
    *,
    dataset_labels: Mapping[str, str],
) -> CatalogIdentityItem:
    exemplars_by_id = {item.asset_id: item for item in identity.exemplars}
    preferred_assets = _preferred_identity_asset_ids(identity)
    resolved_preview = _resolve_identity_exemplar(identity, candidate_ids=preferred_assets)
    preview_url = resolved_preview[1].public_item.preview_url if resolved_preview is not None else None
    thumbnail_url = resolved_preview[1].public_item.thumbnail_url if resolved_preview is not None else None
    enrollment_candidates = [
        _build_identify_exemplar(asset_id, exemplars_by_id, fallback_usage="enrollment_candidate")
        for asset_id in identity.enrollment_candidates
    ]
    probe_candidates = [
        _build_identify_exemplar(asset_id, exemplars_by_id, fallback_usage="probe_candidate")
        for asset_id in identity.probe_candidates
    ]
    recommended_enrollment_capture = None
    if identity.recommended_enrollment_asset_id:
        recommended_enrollment_capture = (
            exemplars_by_id.get(identity.recommended_enrollment_asset_id).capture
            if exemplars_by_id.get(identity.recommended_enrollment_asset_id) is not None
            else None
        )
    recommended_probe_capture = None
    if identity.recommended_probe_asset_id:
        recommended_probe_capture = (
            exemplars_by_id.get(identity.recommended_probe_asset_id).capture
            if exemplars_by_id.get(identity.recommended_probe_asset_id) is not None
            else None
        )
    return CatalogIdentityItem(
        identity_id=identity.identity_id,
        dataset=identity.dataset,
        dataset_label=dataset_labels.get(identity.dataset, identity.dataset),
        display_name=identity.display_name,
        subject_id=_normalize_subject_id(identity.subject_id),
        gallery_role=identity.gallery_role,
        tags=list(identity.tags),
        is_demo_safe=bool(identity.is_demo_safe),
        enrollment_candidates=enrollment_candidates,
        probe_candidates=probe_candidates,
        preview_url=preview_url,
        thumbnail_url=thumbnail_url,
        recommended_enrollment_asset_id=identity.recommended_enrollment_asset_id,
        recommended_probe_asset_id=identity.recommended_probe_asset_id,
        recommended_enrollment_capture=recommended_enrollment_capture,
        recommended_probe_capture=recommended_probe_capture,
    )


def _resolve_identity_exemplar(
    identity: CatalogIdentityRecordModel,
    *,
    candidate_ids: list[str],
) -> tuple[CatalogAssetModel, BrowserAssetRecord] | None:
    exemplars_by_id = {item.asset_id: item for item in identity.exemplars}
    for asset_id in candidate_ids:
        exemplar = exemplars_by_id.get(asset_id)
        if exemplar is None:
            continue
        browser_record = _resolve_browser_record_for_catalog_asset(exemplar)
        if browser_record is not None:
            return exemplar, browser_record
    return None


def _build_demo_identity_record(
    identity: CatalogIdentityRecordModel,
    *,
    dataset_labels: Mapping[str, str],
) -> IdentifyDemoIdentityRecord | None:
    preferred_assets = _preferred_identity_asset_ids(identity)
    resolved = _resolve_identity_exemplar(identity, candidate_ids=preferred_assets)
    if resolved is None:
        return None

    exemplar, browser_record = resolved
    capture = exemplar.capture or browser_record.public_item.capture or "plain"
    public_item = CatalogIdentifyDemoIdentity(
        id=identity.identity_id,
        dataset=identity.dataset,
        dataset_label=dataset_labels.get(identity.dataset, identity.dataset),
        display_label=identity.display_name,
        capture=capture,
        thumbnail_url=browser_record.public_item.thumbnail_url,
        preview_url=browser_record.public_item.preview_url,
        subject_id=_normalize_subject_id(identity.subject_id),
        gallery_role=identity.gallery_role,
        tags=list(identity.tags),
        recommended_enrollment_asset_id=identity.recommended_enrollment_asset_id,
        recommended_probe_asset_id=identity.recommended_probe_asset_id,
    )
    return IdentifyDemoIdentityRecord(
        public_item=public_item,
        enrollment_asset_path=browser_record.asset_paths["preview"],
        enrollment_capture=str(capture),
    )


def _build_identity_seed_record(
    identity: CatalogIdentityRecordModel,
    *,
    dataset_labels: Mapping[str, str],
) -> IdentifySeedIdentityRecord | None:
    public_item = _build_identity_item(identity, dataset_labels=dataset_labels)
    exemplars_by_id = {item.asset_id: item for item in identity.exemplars}
    enrollment_asset_ids = [
        asset_id
        for asset_id in [
            identity.recommended_enrollment_asset_id,
            *identity.enrollment_candidates,
            identity.recommended_probe_asset_id,
            *identity.probe_candidates,
        ]
        if asset_id
    ]

    for asset_id in enrollment_asset_ids:
        exemplar = exemplars_by_id.get(asset_id)
        if exemplar is None:
            continue
        try:
            enrollment_asset_path = _resolve_catalog_asset_path(exemplar)
        except CatalogArtifactError:
            continue
        capture = exemplar.capture or public_item.recommended_enrollment_capture or "plain"
        return IdentifySeedIdentityRecord(
            public_item=public_item,
            enrollment_asset_path=enrollment_asset_path,
            enrollment_capture=capture,
        )

    return None


def _build_probe_case_record_from_scenario(
    scenario: CatalogIdentifyScenarioRecordModel,
    *,
    dataset_labels: Mapping[str, str],
    demo_identity_records: Mapping[str, IdentifyDemoIdentityRecord],
) -> IdentifyProbeCaseRecord | None:
    browser_record = _resolve_browser_record_for_catalog_asset(scenario.probe_asset)
    if browser_record is None:
        return None

    retrieval_method, rerank_method = _resolve_identification_methods(scenario.recommended_method)
    expected_identity = demo_identity_records.get(scenario.expected_identity_id or "")
    public_item = CatalogIdentifyProbeCase(
        id=scenario.scenario_id,
        title=scenario.title,
        description=scenario.description,
        dataset=scenario.dataset,
        dataset_label=dataset_labels.get(scenario.dataset, scenario.dataset),
        capture=scenario.probe_asset.capture or browser_record.public_item.capture,
        difficulty=scenario.difficulty,
        probe_thumbnail_url=browser_record.public_item.thumbnail_url,
        probe_preview_url=browser_record.public_item.preview_url,
        probe_asset_url=browser_record.public_item.preview_url,
        expected_outcome=_expected_outcome_for_scenario(scenario),
        expected_top_identity_id=scenario.expected_identity_id,
        expected_top_identity_label=expected_identity.public_item.display_label if expected_identity else None,
        recommended_retrieval_method=retrieval_method,
        recommended_rerank_method=rerank_method,
        recommended_shortlist_size=max(5, len(scenario.enrollment_identity_ids) * 5),
        scenario_type=scenario.scenario_type,
        tags=list(scenario.tags),
    )
    return IdentifyProbeCaseRecord(
        public_item=public_item,
        probe_asset_path=browser_record.asset_paths["preview"],
    )


def _build_identity_probe_case_record(
    identity: CatalogIdentityRecordModel,
    *,
    dataset_labels: Mapping[str, str],
) -> IdentifyProbeCaseRecord | None:
    preferred_assets = [
        asset_id
        for asset_id in [
            identity.recommended_probe_asset_id,
            *identity.probe_candidates,
            identity.recommended_enrollment_asset_id,
        ]
        if asset_id
    ]
    resolved = _resolve_identity_exemplar(identity, candidate_ids=preferred_assets)
    if resolved is None:
        return None

    exemplar, browser_record = resolved
    retrieval_method, rerank_method = _resolve_identification_methods(None)
    public_item = CatalogIdentifyProbeCase(
        id=f"probe_{identity.identity_id}",
        title=f"Probe {identity.display_name}",
        description="Run 1:N against the seeded demo gallery using the recommended probe exemplar for this identity.",
        dataset=identity.dataset,
        dataset_label=dataset_labels.get(identity.dataset, identity.dataset),
        capture=exemplar.capture or browser_record.public_item.capture,
        difficulty="easy",
        probe_thumbnail_url=browser_record.public_item.thumbnail_url,
        probe_preview_url=browser_record.public_item.preview_url,
        probe_asset_url=browser_record.public_item.preview_url,
        expected_outcome="match",
        expected_top_identity_id=identity.identity_id,
        expected_top_identity_label=identity.display_name,
        recommended_retrieval_method=retrieval_method,
        recommended_rerank_method=rerank_method,
        recommended_shortlist_size=10,
        scenario_type="positive_identification",
        tags=list(identity.tags),
    )
    return IdentifyProbeCaseRecord(
        public_item=public_item,
        probe_asset_path=browser_record.asset_paths["preview"],
    )


def _build_additional_demo_probe_case_records(
    identity: CatalogIdentityRecordModel,
    *,
    dataset_labels: Mapping[str, str],
    existing_probe_asset_ids: set[str],
    limit: int,
) -> list[IdentifyProbeCaseRecord]:
    if limit <= 0:
        return []

    exemplars_by_id = {item.asset_id: item for item in identity.exemplars}
    same_subject_records: list[tuple[str | None, BrowserAssetRecord]] = []
    seen_browser_asset_ids: set[str] = set()
    for asset_id in _preferred_identity_asset_ids(identity):
        exemplar = exemplars_by_id.get(asset_id)
        if exemplar is None:
            continue
        browser_record = _resolve_browser_record_for_catalog_asset(exemplar)
        if browser_record is None:
            continue
        browser_asset_id = browser_record.public_item.asset_id
        if browser_asset_id in existing_probe_asset_ids or browser_asset_id in seen_browser_asset_ids:
            continue
        same_subject_records.append((exemplar.capture, browser_record))
        seen_browser_asset_ids.add(browser_asset_id)

    try:
        browser_state = _get_browser_dataset_state(identity.dataset)
    except CatalogApiError:
        browser_state = None

    other_subject_records: list[BrowserAssetRecord] = []
    if browser_state is not None:
        identity_subject_id = _normalize_subject_id(identity.subject_id)
        for record in browser_state.items:
            browser_asset_id = record.public_item.asset_id
            if browser_asset_id in existing_probe_asset_ids or browser_asset_id in seen_browser_asset_ids:
                continue
            if _normalize_subject_id(record.public_item.subject_id) == identity_subject_id:
                same_subject_records.append((record.public_item.capture, record))
                seen_browser_asset_ids.add(browser_asset_id)
                continue
            other_subject_records.append(record)
            seen_browser_asset_ids.add(browser_asset_id)

    retrieval_method, rerank_method = _resolve_identification_methods(None)
    additional_records: list[IdentifyProbeCaseRecord] = []

    def append_record(
        *,
        case_suffix: str,
        title: str,
        description: str,
        scenario_type: str,
        difficulty: str,
        browser_record: BrowserAssetRecord,
        capture: str | None,
        expected_identity_id: str | None,
        expected_label: str | None,
        tags: list[str],
    ) -> None:
        if len(additional_records) >= limit:
            return
        additional_records.append(
            IdentifyProbeCaseRecord(
                public_item=CatalogIdentifyProbeCase(
                    id=f"probe_{identity.identity_id}_{case_suffix}",
                    title=title,
                    description=description,
                    dataset=identity.dataset,
                    dataset_label=dataset_labels.get(identity.dataset, identity.dataset),
                    capture=capture or browser_record.public_item.capture,
                    difficulty=difficulty,
                    probe_thumbnail_url=browser_record.public_item.thumbnail_url,
                    probe_preview_url=browser_record.public_item.preview_url,
                    probe_asset_url=browser_record.public_item.preview_url,
                    expected_outcome="match" if expected_identity_id else "no_match",
                    expected_top_identity_id=expected_identity_id,
                    expected_top_identity_label=expected_label,
                    recommended_retrieval_method=retrieval_method,
                    recommended_rerank_method=rerank_method,
                    recommended_shortlist_size=10,
                    scenario_type=scenario_type,
                    tags=tags,
                ),
                probe_asset_path=browser_record.asset_paths["preview"],
            )
        )

    if same_subject_records:
        capture, browser_record = same_subject_records[0]
        append_record(
            case_suffix="positive_followup",
            title=f"Positive identification follow-up for {identity.display_name}",
            description="A second positive 1:N walkthrough that keeps the gallery fixed while varying the probe asset.",
            scenario_type="positive_identification",
            difficulty="easy",
            browser_record=browser_record,
            capture=capture,
            expected_identity_id=identity.identity_id,
            expected_label=identity.display_name,
            tags=[identity.dataset, "positive_identification", "demo_safe", "followup"],
        )

    if len(same_subject_records) > 1:
        capture, browser_record = same_subject_records[1]
        append_record(
            case_suffix="harder_positive",
            title=f"Harder positive identification for {identity.display_name}",
            description="A higher-friction positive search that still should recover the enrolled identity from the curated gallery.",
            scenario_type="difficult_identification",
            difficulty="hard",
            browser_record=browser_record,
            capture=capture,
            expected_identity_id=identity.identity_id,
            expected_label=identity.display_name,
            tags=[identity.dataset, "difficult_identification", "demo_safe", "positive_harder"],
        )

    for index, browser_record in enumerate(other_subject_records, start=1):
        append_record(
            case_suffix=f"no_match_{index}",
            title=f"No-match validation probe {index}",
            description="A curated negative path that keeps the gallery stable and swaps in a different-subject probe.",
            scenario_type="no_match",
            difficulty="hard" if index == 1 else "challenging",
            browser_record=browser_record,
            capture=browser_record.public_item.capture,
            expected_identity_id=None,
            expected_label=None,
            tags=[identity.dataset, "no_match", "demo_safe", "negative_path"],
        )
        if len(additional_records) >= limit:
            break

    return additional_records


def _read_json(path: Path, *, missing_message: str, invalid_message: str) -> dict[str, Any]:
    if not path.is_file():
        raise CatalogArtifactError(missing_message.format(path=path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogArtifactError(invalid_message.format(path=path)) from exc
    except OSError as exc:
        raise CatalogArtifactError(f"Could not read JSON artifact: {path}") from exc


def _load_catalog_payload() -> CatalogPayloadModel:
    payload = _read_json(
        _catalog_path(),
        missing_message="Catalog artifact is missing: {path}",
        invalid_message="Catalog artifact is not valid JSON: {path}",
    )
    try:
        return CatalogPayloadModel.model_validate(payload)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        raise CatalogArtifactError(f"Catalog artifact failed validation: {message}") from exc


def _load_ui_assets_registry_payload() -> UiAssetsRegistryPayloadModel:
    payload = _read_json(
        _ui_assets_registry_path(),
        missing_message="UI assets registry is missing: {path}",
        invalid_message="UI assets registry is not valid JSON: {path}",
    )
    try:
        return UiAssetsRegistryPayloadModel.model_validate(payload)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        raise CatalogArtifactError(f"UI assets registry failed validation: {message}") from exc


@lru_cache(maxsize=1)
def _get_catalog_state() -> CatalogState:
    payload = _load_catalog_payload()
    dataset_labels: Dict[str, str] = {}
    catalog_datasets: set[str] = set()
    source_dataset_records: list[CatalogSourceDatasetModel] = []

    for raw_dataset in payload.source_datasets:
        try:
            dataset = CatalogSourceDatasetModel.model_validate(raw_dataset)
        except ValidationError:
            continue
        dataset_labels[dataset.dataset] = dataset.dataset_label
        catalog_datasets.add(dataset.dataset)
        source_dataset_records.append(dataset)

    try:
        public_demo_cases = demo_store.load_demo_cases().cases
    except demo_store.DemoCatalogError as exc:
        raise CatalogArtifactError(str(exc)) from exc

    raw_verify_cases: Dict[str, CatalogVerifyCaseModel] = {}
    for raw_case in payload.verify_cases:
        try:
            case = CatalogVerifyCaseModel.model_validate(raw_case)
        except ValidationError as exc:
            logger.info("Skipping invalid verify case while building catalog API state: %s", exc)
            continue
        raw_verify_cases[case.case_id] = case

    verify_items: list[CatalogVerifyCaseItem] = []
    verify_details: Dict[str, CatalogVerifyCaseDetail] = {}

    for public_case in public_demo_cases:
        raw_case = raw_verify_cases.get(public_case.id)
        if raw_case is None:
            logger.warning("Skipping public demo case %s because raw catalog details could not be resolved.", public_case.id)
            continue

        item = CatalogVerifyCaseItem(
            case_id=raw_case.case_id,
            title=raw_case.title,
            description=raw_case.description,
            dataset=raw_case.dataset,
            dataset_label=dataset_labels.get(raw_case.dataset, raw_case.dataset),
            split=raw_case.split,
            difficulty=raw_case.difficulty,
            case_type=raw_case.case_type,
            ground_truth=raw_case.ground_truth,
            recommended_method=public_case.recommended_method,
            capture_a=raw_case.capture_a,
            capture_b=raw_case.capture_b,
            modality_relation=raw_case.modality_relation,
            tags=list(raw_case.tags),
            selection_policy=raw_case.selection_policy,
            selection_reason=raw_case.selection_reason,
            image_a_url=public_case.image_a_url,
            image_b_url=public_case.image_b_url,
            availability_status=raw_case.availability_status,
            asset_a_id=public_case.asset_a_id,
            asset_b_id=public_case.asset_b_id,
            evidence_quality=public_case.evidence_quality,
        )
        verify_items.append(item)
        verify_details[item.case_id] = CatalogVerifyCaseDetail(
            **item.model_dump(),
            benchmark_context=normalize_benchmark_context(raw_case.benchmark_context, split=raw_case.split),
            traceability_summary=_build_verify_traceability(raw_case),
            additional_notes=_build_verify_notes(raw_case),
        )

    case_evidence_by_dataset: Dict[str, list[Any]] = {}
    for item in verify_items:
        if item.evidence_quality is None:
            continue
        case_evidence_by_dataset.setdefault(item.dataset, []).append(item.evidence_quality)

    built_case_counts = Counter(item.dataset for item in verify_items)
    dataset_demo_health: Dict[str, CatalogDatasetDemoHealth] = {}
    for dataset in source_dataset_records:
        demo_health = build_dataset_demo_health_summary(
            dataset.verify_selection_diagnostics,
            case_evidence=case_evidence_by_dataset.get(dataset.dataset, []),
            built_case_count=built_case_counts.get(dataset.dataset, 0),
        )
        if demo_health is not None:
            dataset_demo_health[dataset.dataset] = demo_health

    for dataset, built_case_count in built_case_counts.items():
        if dataset in dataset_demo_health:
            continue
        demo_health = build_dataset_demo_health_summary(
            [],
            case_evidence=case_evidence_by_dataset.get(dataset, []),
            built_case_count=built_case_count,
        )
        if demo_health is not None:
            dataset_demo_health[dataset] = demo_health

    catalog_build_health = build_catalog_build_health_summary(
        payload.metadata.get("catalog_build_health"),
        case_evidence=[
            item.evidence_quality
            for item in verify_items
            if item.evidence_quality is not None
        ],
        dataset_demo_health=dataset_demo_health,
    )

    identity_items: list[CatalogIdentityItem] = []
    raw_identity_records: list[CatalogIdentityRecordModel] = []
    for raw_identity in payload.identify_gallery.identities:
        try:
            identity = CatalogIdentityRecordModel.model_validate(raw_identity)
        except ValidationError as exc:
            logger.info("Skipping invalid identify-gallery record while building catalog API state: %s", exc)
            continue
        raw_identity_records.append(identity)
        identity_items.append(_build_identity_item(identity, dataset_labels=dataset_labels))

    identity_items.sort(key=lambda item: (item.dataset_label.lower(), item.display_name.lower(), item.identity_id))

    identity_seed_records: Dict[str, IdentifySeedIdentityRecord] = {}
    for identity in raw_identity_records:
        record = _build_identity_seed_record(identity, dataset_labels=dataset_labels)
        if record is None:
            logger.info("Skipping seedable identify identity %s because an enrollment asset could not be resolved.", identity.identity_id)
            continue
        identity_seed_records[identity.identity_id] = record

    demo_identity_records: Dict[str, IdentifyDemoIdentityRecord] = {}
    for identity in raw_identity_records:
        if not bool(identity.is_demo_safe):
            continue
        record = _build_demo_identity_record(identity, dataset_labels=dataset_labels)
        if record is None:
            logger.info("Skipping demo-safe identify identity %s because preview assets could not be resolved.", identity.identity_id)
            continue
        demo_identity_records[identity.identity_id] = record

    probe_case_records: Dict[str, IdentifyProbeCaseRecord] = {}
    for raw_scenario in payload.identify_gallery.demo_scenarios:
        try:
            scenario = CatalogIdentifyScenarioRecordModel.model_validate(raw_scenario)
        except ValidationError as exc:
            logger.info("Skipping invalid identify-gallery probe scenario while building catalog API state: %s", exc)
            continue

        if not bool(scenario.is_demo_safe):
            continue
        if any(identity_id not in demo_identity_records for identity_id in scenario.enrollment_identity_ids):
            logger.info(
                "Skipping identify-gallery probe scenario %s because one or more enrollment identities are unavailable.",
                scenario.scenario_id,
            )
            continue

        record = _build_probe_case_record_from_scenario(
            scenario,
            dataset_labels=dataset_labels,
            demo_identity_records=demo_identity_records,
        )
        if record is None:
            logger.info("Skipping identify-gallery probe scenario %s because preview assets could not be resolved.", scenario.scenario_id)
            continue
        probe_case_records[record.public_item.id] = record

    covered_identity_ids = {
        record.public_item.expected_top_identity_id
        for record in probe_case_records.values()
        if record.public_item.expected_top_identity_id
    }
    for identity in raw_identity_records:
        if not bool(identity.is_demo_safe) or identity.identity_id not in demo_identity_records:
            continue
        if identity.identity_id in covered_identity_ids:
            continue
        record = _build_identity_probe_case_record(identity, dataset_labels=dataset_labels)
        if record is None:
            continue
        probe_case_records[record.public_item.id] = record

    desired_probe_case_count = 5
    if len(probe_case_records) < desired_probe_case_count:
        existing_probe_asset_ids = {
            asset_id
            for record in probe_case_records.values()
            for asset_id in [_catalog_asset_id_from_api_url(record.public_item.probe_preview_url)]
            if asset_id
        }
        for identity in raw_identity_records:
            if not bool(identity.is_demo_safe) or identity.identity_id not in demo_identity_records:
                continue
            needed = desired_probe_case_count - len(probe_case_records)
            if needed <= 0:
                break
            additional_records = _build_additional_demo_probe_case_records(
                identity,
                dataset_labels=dataset_labels,
                existing_probe_asset_ids=existing_probe_asset_ids,
                limit=needed,
            )
            for record in additional_records:
                probe_case_records.setdefault(record.public_item.id, record)
                asset_id = _catalog_asset_id_from_api_url(record.public_item.probe_preview_url)
                if asset_id:
                    existing_probe_asset_ids.add(asset_id)

    demo_identity_items = sorted(
        (record.public_item for record in demo_identity_records.values()),
        key=lambda item: (item.dataset_label.lower(), item.display_label.lower(), item.id),
    )
    probe_case_items = sorted(
        (record.public_item for record in probe_case_records.values()),
        key=_probe_case_sort_key,
    )

    return CatalogState(
        dataset_labels=dataset_labels,
        catalog_datasets=catalog_datasets,
        verify_items=verify_items,
        verify_details=verify_details,
        catalog_build_health=catalog_build_health,
        dataset_demo_health=dataset_demo_health,
        identity_items=identity_items,
        identity_seed_records=identity_seed_records,
        demo_identity_items=demo_identity_items,
        probe_case_items=probe_case_items,
        demo_identity_records=demo_identity_records,
        probe_case_records=probe_case_records,
    )


@lru_cache(maxsize=1)
def _get_ui_assets_registry_entries() -> Dict[str, UiAssetsRegistryEntryModel]:
    payload = _load_ui_assets_registry_payload()
    return {entry.dataset: entry for entry in payload.datasets}


@lru_cache(maxsize=16)
def _load_ui_assets_index_payload(dataset: str) -> UiAssetsIndexPayloadModel:
    entry = _get_ui_assets_registry_entries().get(dataset)
    if entry is None:
        raise CatalogDatasetNotFoundError(f"Browser assets are not available for dataset {dataset!r}.")

    path = _resolve_browser_index_path(entry.index_path)
    payload = _read_json(
        path,
        missing_message="UI assets index is missing: {path}",
        invalid_message="UI assets index is not valid JSON: {path}",
    )
    try:
        index = UiAssetsIndexPayloadModel.model_validate(payload)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        raise CatalogArtifactError(f"UI assets index failed validation for dataset {dataset!r}: {message}") from exc

    if index.dataset != dataset:
        raise CatalogArtifactError(
            f"UI assets index dataset mismatch: expected {dataset!r}, found {index.dataset!r}."
        )
    return index


@lru_cache(maxsize=16)
def _load_ui_assets_validation_report(dataset: str) -> UiAssetsValidationReportModel:
    entry = _get_ui_assets_registry_entries().get(dataset)
    if entry is None:
        raise CatalogDatasetNotFoundError(f"Browser assets are not available for dataset {dataset!r}.")

    path = _resolve_browser_index_path(entry.validation_report_path)
    payload = _read_json(
        path,
        missing_message="UI assets validation report is missing: {path}",
        invalid_message="UI assets validation report is not valid JSON: {path}",
    )
    try:
        report = UiAssetsValidationReportModel.model_validate(payload)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        raise CatalogArtifactError(
            f"UI assets validation report failed validation for dataset {dataset!r}: {message}"
        ) from exc

    if report.dataset != dataset:
        raise CatalogArtifactError(
            f"UI assets validation report dataset mismatch: expected {dataset!r}, found {report.dataset!r}."
        )
    return report


@lru_cache(maxsize=16)
def _get_browser_dataset_state(dataset: str) -> BrowserDatasetState:
    index = _load_ui_assets_index_payload(dataset)
    report = _load_ui_assets_validation_report(dataset)

    items: list[BrowserAssetRecord] = []
    items_by_id: Dict[str, BrowserAssetRecord] = {}
    items_by_source_key: Dict[str, BrowserAssetRecord] = {}
    exclusions: Dict[str, str] = {}

    for index_position, raw_item in enumerate(index.items):
        item_key = f"<index:{index_position}>"
        if isinstance(raw_item, dict) and raw_item.get("asset_id"):
            item_key = str(raw_item["asset_id"])

        try:
            item = UiAssetsIndexItemModel.model_validate(raw_item)
        except ValidationError as exc:
            exclusions[item_key] = f"item failed validation: {exc.errors()[0]['msg'] if exc.errors() else exc}"
            continue

        if str(item.availability_status) != "available":
            exclusions[item.asset_id] = f"availability_status={item.availability_status!r}"
            continue

        try:
            thumbnail_path = _resolve_browser_asset_path(dataset, item.thumbnail_path)
            preview_path = _resolve_browser_asset_path(dataset, item.preview_path)
        except CatalogApiError as exc:
            exclusions[item.asset_id] = str(exc)
            continue

        public_item = CatalogBrowserItem(
            asset_id=item.asset_id,
            dataset=item.dataset,
            split=item.split,
            subject_id=_normalize_optional_text(item.subject_id),
            finger=_normalize_optional_text(item.finger),
            capture=_normalize_optional_text(item.capture),
            modality=_normalize_optional_text(item.modality),
            ui_eligible=bool(item.ui_eligible),
            selection_reason=item.selection_reason,
            selection_policy=item.selection_policy,
            thumbnail_url=_safe_browser_asset_url(dataset, item.asset_id, "thumbnail"),
            preview_url=_safe_browser_asset_url(dataset, item.asset_id, "preview"),
            availability_status=item.availability_status,
            original_dimensions=_asset_dimensions(item.original_dimensions),
            thumbnail_dimensions=_asset_dimensions(item.thumbnail_dimensions),
            preview_dimensions=_asset_dimensions(item.preview_dimensions),
        )
        record = BrowserAssetRecord(
            public_item=public_item,
            asset_paths={"thumbnail": thumbnail_path, "preview": preview_path},
            index_position=index_position,
            split_subject_sort_key=(
                item.split,
                _normalize_subject_id(item.subject_id),
                _normalize_optional_text(item.finger) or "",
                item.asset_id,
            ),
        )
        items.append(record)
        items_by_id[item.asset_id] = record
        for lookup_key in _browser_lookup_keys_for_ui_item(item):
            items_by_source_key.setdefault(lookup_key, record)

    if exclusions:
        logger.info("Excluded %s ui_assets item(s) for dataset %s: %s", len(exclusions), dataset, exclusions)

    return BrowserDatasetState(
        dataset=index.dataset,
        dataset_label=index.dataset_label,
        selection_policy=index.selection_policy,
        validation_status=report.validation_status or index.validation_status,
        generated_at=index.generated_at,
        generator_version=index.generator_version,
        warning_count=max(int(report.excluded_records or 0), 0),
        summary=dict(index.summary),
        items=items,
        items_by_id=items_by_id,
        items_by_source_key=items_by_source_key,
        excluded_asset_reasons=exclusions,
    )


def clear_catalog_store_cache() -> None:
    _get_catalog_state.cache_clear()
    _get_ui_assets_registry_entries.cache_clear()
    _load_ui_assets_index_payload.cache_clear()
    _load_ui_assets_validation_report.cache_clear()
    _get_browser_dataset_state.cache_clear()


def load_catalog_datasets() -> CatalogDatasetsResponse:
    catalog_state = _get_catalog_state()
    try:
        registry_entries = _get_ui_assets_registry_entries()
    except CatalogArtifactError:
        if _ui_assets_registry_path().is_file():
            raise
        logger.warning(
            "UI assets registry is missing at %s. Returning catalog-backed datasets without browser assets.",
            _ui_assets_registry_path(),
        )
        registry_entries = {}
    verify_counts = Counter(item.dataset for item in catalog_state.verify_items)
    identify_counts = Counter(item.dataset for item in catalog_state.identity_items)

    dataset_labels = dict(catalog_state.dataset_labels)
    for dataset in registry_entries:
        if dataset not in dataset_labels:
            dataset_labels[dataset] = _load_ui_assets_index_payload(dataset).dataset_label

    all_datasets = set(dataset_labels) | set(registry_entries) | set(verify_counts) | set(identify_counts)
    summaries: list[CatalogDatasetSummary] = []

    for dataset in sorted(all_datasets, key=lambda key: (dataset_labels.get(key, key).lower(), key)):
        verify_count = verify_counts.get(dataset, 0)
        identify_count = identify_counts.get(dataset, 0)
        browser_entry = registry_entries.get(dataset)
        has_browser_assets = browser_entry is not None and browser_entry.item_count > 0
        available_features = []
        if verify_count > 0:
            available_features.append("verify_cases")
        if identify_count > 0:
            available_features.append("identify_gallery")
        if has_browser_assets:
            available_features.append("dataset_browser")

        selection_policy = None
        if browser_entry is not None:
            selection_policy = browser_entry.selection_policy
        else:
            for item in catalog_state.verify_items:
                if item.dataset == dataset:
                    selection_policy = item.selection_policy
                    break

        summaries.append(
            CatalogDatasetSummary(
                dataset=dataset,
                dataset_label=dataset_labels.get(dataset, dataset),
                has_verify_cases=verify_count > 0,
                has_identify_gallery=identify_count > 0,
                has_browser_assets=has_browser_assets,
                verify_case_count=verify_count,
                identify_identity_count=identify_count,
                browser_item_count=browser_entry.item_count if browser_entry is not None else 0,
                browser_validation_status=browser_entry.validation_status if browser_entry is not None else None,
                selection_policy=selection_policy,
                available_features=available_features,
                demo_health=catalog_state.dataset_demo_health.get(dataset),
            )
        )

    return CatalogDatasetsResponse(items=summaries, catalog_build_health=catalog_state.catalog_build_health)


def load_catalog_verify_cases(
    *,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    difficulty: Optional[str] = None,
    case_type: Optional[str] = None,
    ground_truth: Optional[str] = None,
    modality_relation: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> CatalogVerifyCasesResponse:
    effective_limit, effective_offset = _normalize_limit_offset(limit, offset, default_limit=DEFAULT_PAGE_LIMIT)
    catalog_state = _get_catalog_state()
    items = catalog_state.verify_items
    filtered = []
    normalized_tag = (_normalize_optional_text(tag) or "").lower()

    for item in items:
        if dataset and not _case_insensitive_equals(item.dataset, dataset):
            continue
        if split and not _case_insensitive_equals(item.split, split):
            continue
        if difficulty and not _case_insensitive_equals(item.difficulty, difficulty):
            continue
        if case_type and not _case_insensitive_equals(item.case_type, case_type):
            continue
        if ground_truth and not _case_insensitive_equals(item.ground_truth, ground_truth):
            continue
        if modality_relation and not _case_insensitive_equals(item.modality_relation, modality_relation):
            continue
        if normalized_tag and normalized_tag not in {entry.lower() for entry in item.tags}:
            continue
        filtered.append(item)

    window, has_more = _paginate(filtered, limit=effective_limit, offset=effective_offset)
    return CatalogVerifyCasesResponse(
        items=window,
        total=len(filtered),
        limit=effective_limit,
        offset=effective_offset,
        has_more=has_more,
        catalog_build_health=catalog_state.catalog_build_health,
    )


def load_catalog_verify_case_detail(case_id: str) -> CatalogVerifyCaseDetail:
    detail = _get_catalog_state().verify_details.get(case_id)
    if detail is None:
        raise CatalogVerifyCaseNotFoundError(f"Verify case {case_id!r} is not available.")
    return detail


def load_catalog_identify_demo_identity_records(
    *,
    dataset: Optional[str] = None,
) -> list[IdentifyDemoIdentityRecord]:
    state = _get_catalog_state()
    records = list(state.demo_identity_records.values())
    if dataset:
        records = [record for record in records if _case_insensitive_equals(record.public_item.dataset, dataset)]
    records.sort(key=lambda record: (record.public_item.dataset_label.lower(), record.public_item.display_label.lower(), record.public_item.id))
    return records


def load_catalog_identify_seed_records(
    *,
    dataset: Optional[str] = None,
    selected_identity_ids: Optional[list[str]] = None,
) -> list[IdentifySeedIdentityRecord]:
    state = _get_catalog_state()
    records = list(state.identity_seed_records.values())
    if dataset:
        records = [record for record in records if _case_insensitive_equals(record.public_item.dataset, dataset)]
    if selected_identity_ids is not None:
        selected_lookup = {str(identity_id).strip() for identity_id in selected_identity_ids if str(identity_id).strip()}
        records = [record for record in records if record.public_item.identity_id in selected_lookup]
    records.sort(key=lambda record: (record.public_item.dataset_label.lower(), record.public_item.display_name.lower(), record.public_item.identity_id))
    return records


def load_catalog_identify_probe_case_records(
    *,
    dataset: Optional[str] = None,
) -> list[IdentifyProbeCaseRecord]:
    state = _get_catalog_state()
    records = list(state.probe_case_records.values())
    if dataset:
        records = [record for record in records if _case_insensitive_equals(record.public_item.dataset, dataset)]
    records.sort(key=lambda record: (record.public_item.dataset_label.lower(), record.public_item.difficulty.lower(), record.public_item.title.lower(), record.public_item.id))
    return records


def load_catalog_identify_gallery(
    *,
    dataset: Optional[str] = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> CatalogIdentifyGalleryResponse:
    effective_limit, effective_offset = _normalize_limit_offset(limit, offset, default_limit=DEFAULT_PAGE_LIMIT)
    items = _get_catalog_state().identity_items
    filtered = [item for item in items if not dataset or _case_insensitive_equals(item.dataset, dataset)]
    window, has_more = _paginate(filtered, limit=effective_limit, offset=effective_offset)
    demo_identity_records = load_catalog_identify_demo_identity_records(dataset=dataset)
    probe_case_records = load_catalog_identify_probe_case_records(dataset=dataset)
    return CatalogIdentifyGalleryResponse(
        items=window,
        demo_identities=[record.public_item for record in demo_identity_records],
        probe_cases=[record.public_item for record in probe_case_records],
        total=len(filtered),
        limit=effective_limit,
        offset=effective_offset,
        has_more=has_more,
        total_probe_cases=len(probe_case_records),
    )


def load_catalog_dataset_browser(
    *,
    dataset: str,
    split: Optional[str] = None,
    capture: Optional[str] = None,
    modality: Optional[str] = None,
    subject_id: Optional[str] = None,
    finger: Optional[str] = None,
    ui_eligible: Optional[bool] = None,
    limit: int = DEFAULT_BROWSER_LIMIT,
    offset: int = 0,
    sort: str = "default",
) -> CatalogDatasetBrowserResponse:
    if sort not in ALLOWED_BROWSER_SORTS:
        raise CatalogInvalidRequestError(
            f"Unsupported sort value {sort!r}. Expected one of: {', '.join(sorted(ALLOWED_BROWSER_SORTS))}."
        )

    effective_limit, effective_offset = _normalize_limit_offset(limit, offset, default_limit=DEFAULT_BROWSER_LIMIT)
    state = _get_browser_dataset_state(dataset)
    filtered = []

    for record in state.items:
        item = record.public_item
        if split and not _case_insensitive_equals(item.split, split):
            continue
        if capture and not _case_insensitive_equals(item.capture, capture):
            continue
        if modality and not _case_insensitive_equals(item.modality, modality):
            continue
        if subject_id and not _case_insensitive_equals(item.subject_id, subject_id):
            continue
        if finger and not _case_insensitive_equals(item.finger, finger):
            continue
        if ui_eligible is not None and bool(item.ui_eligible) != bool(ui_eligible):
            continue
        filtered.append(record)

    if sort == "split_subject_asset":
        filtered.sort(key=lambda record: record.split_subject_sort_key)
    else:
        filtered.sort(key=lambda record: record.index_position)

    window, has_more = _paginate(filtered, limit=effective_limit, offset=effective_offset)
    return CatalogDatasetBrowserResponse(
        dataset=state.dataset,
        dataset_label=state.dataset_label,
        selection_policy=state.selection_policy,
        validation_status=state.validation_status,
        total=len(filtered),
        limit=effective_limit,
        offset=effective_offset,
        has_more=has_more,
        generated_at=state.generated_at,
        generator_version=state.generator_version,
        warning_count=state.warning_count,
        summary=dict(state.summary),
        items=[record.public_item for record in window],
    )


def resolve_catalog_browser_asset_path(dataset: str, asset_id: str, variant: str) -> Path:
    normalized_variant = str(variant).strip().lower()
    if normalized_variant not in {"thumbnail", "preview"}:
        raise CatalogInvalidRequestError(
            f"Unknown browser asset variant: {variant!r}. Expected 'thumbnail' or 'preview'."
        )

    state = _get_browser_dataset_state(dataset)
    record = state.items_by_id.get(asset_id)
    if record is None:
        if asset_id in state.excluded_asset_reasons:
            raise CatalogBrowserAssetNotFoundError(
                f"Browser asset {asset_id!r} is not available: {state.excluded_asset_reasons[asset_id]}."
            )
        raise CatalogBrowserAssetNotFoundError(f"Unknown browser asset {asset_id!r} for dataset {dataset!r}.")

    path = record.asset_paths[normalized_variant]
    if not path.is_file():
        raise CatalogBrowserAssetNotFoundError(
            f"Browser asset file for dataset {dataset!r}, asset {asset_id!r}, and variant {normalized_variant!r} is missing."
        )
    return path
