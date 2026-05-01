from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apps.api.benchmark_meta import benchmark_method_to_canonical, normalize_benchmark_context
from apps.api.public_health import build_catalog_build_health_summary, build_evidence_quality_summary
from apps.api.schemas import CatalogBuildHealthSummary, DemoCase, DemoCasesResponse, MatchMethod

ROOT = Path(__file__).resolve().parents[2]
SAMPLES_ROOT = ROOT / "data" / "samples"
CATALOG_PATH = SAMPLES_ROOT / "catalog.json"
ASSETS_ROOT = SAMPLES_ROOT / "assets"

logger = logging.getLogger(__name__)

SPLIT_LABELS = {
    "val": "Validation",
    "test": "Test",
    "train": "Train",
}
GROUND_TRUTH_LABELS = {
    "match": (1, "Genuine pair"),
    "non_match": (0, "Impostor pair"),
}
DIFFICULTY_ORDER = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
    "challenging": 3,
}


class DemoStoreError(RuntimeError):
    pass


class DemoCatalogError(DemoStoreError):
    pass


class DemoCaseNotFoundError(DemoStoreError):
    pass


class DemoInvalidSlotError(DemoStoreError):
    pass


class DemoAssetResolutionError(DemoStoreError):
    pass


class CatalogSourceDatasetModel(BaseModel):
    dataset: str
    dataset_label: str
    verify_selection_diagnostics: Optional[list[dict[str, Any]]] = None

    model_config = ConfigDict(extra="allow")


class CatalogAssetModel(BaseModel):
    asset_id: str
    path: str
    relative_path: Optional[str] = None
    availability_status: str
    capture: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class CatalogVerifyCaseModel(BaseModel):
    case_id: str
    title: str
    description: str
    dataset: str
    split: str
    case_type: str
    difficulty: str
    ground_truth: str
    recommended_method: str
    capture_a: str
    capture_b: str
    image_a: CatalogAssetModel
    image_b: CatalogAssetModel
    is_demo_safe: bool
    availability_status: str
    selection_reason: str
    selection_policy: str
    tags: list[str] = Field(default_factory=list)
    modality_relation: Optional[str] = None
    benchmark_context: Optional[dict[str, Any]] = None
    selection_diagnostics: Optional[dict[str, Any]] = None
    priority: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class CatalogPayloadModel(BaseModel):
    verify_cases: list[Any]
    source_datasets: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class ResolvedDemoRecord:
    public_case: DemoCase
    asset_paths: Dict[str, Path]
    sort_key: tuple[Any, ...]


@dataclass(frozen=True)
class DemoStoreState:
    cases_by_id: Dict[str, ResolvedDemoRecord] = field(default_factory=dict)
    excluded_case_reasons: Dict[str, str] = field(default_factory=dict)
    catalog_build_health: Optional[CatalogBuildHealthSummary] = None


def _repo_root() -> Path:
    override = os.getenv("FPBENCH_DEMO_ROOT")
    return Path(override).resolve() if override else ROOT.resolve()


def _catalog_path() -> Path:
    override = os.getenv("FPBENCH_DEMO_CATALOG_PATH")
    return Path(override).resolve() if override else CATALOG_PATH.resolve()


def _assets_root() -> Path:
    override = os.getenv("FPBENCH_DEMO_ASSETS_ROOT")
    return Path(override).resolve() if override else ASSETS_ROOT.resolve()


def _path_is_under(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _looks_absolute(path_str: str) -> bool:
    return PurePosixPath(path_str).is_absolute() or PureWindowsPath(path_str).is_absolute()


def _safe_asset_url(case_id: str, slot: str) -> str:
    return f"/api/demo/cases/{case_id}/{slot}"


def _split_label(split: str) -> str:
    return SPLIT_LABELS.get(split, split.replace("_", " ").title())


def _ground_truth_payload(value: str) -> tuple[int, str]:
    label, text = GROUND_TRUTH_LABELS.get(value, (-1, value.replace("_", " ").title()))
    if label == -1:
        raise DemoCatalogError(f"Unsupported ground truth value: {value}")
    return label, text


def _is_case_disabled(case: CatalogVerifyCaseModel) -> bool:
    extras = case.model_extra or {}
    if bool(extras.get("hidden")):
        return True
    if bool(extras.get("disabled")):
        return True
    if extras.get("enabled") is False:
        return True
    return False


def _normalize_catalog_asset_path(path_str: str) -> Path:
    if not path_str:
        raise DemoAssetResolutionError("Demo asset path is missing from the catalog.")

    normalized = str(path_str).replace("\\", "/").strip()
    if not normalized:
        raise DemoAssetResolutionError("Demo asset path is empty.")
    if _looks_absolute(normalized):
        raise DemoAssetResolutionError("Absolute demo asset paths are not allowed.")

    logical = PurePosixPath(normalized)
    if any(part == ".." for part in logical.parts):
        raise DemoAssetResolutionError("Demo asset path traversal is not allowed.")

    root = _repo_root()
    assets_root = _assets_root()
    if normalized.startswith("data/samples/assets/"):
        candidate = root.joinpath(*logical.parts)
    else:
        candidate = assets_root.joinpath(*logical.parts)

    resolved = candidate.resolve()
    if not _path_is_under(assets_root, resolved):
        raise DemoAssetResolutionError("Resolved demo asset path falls outside the allowed asset root.")
    return resolved


def _resolve_catalog_asset(asset: CatalogAssetModel) -> Path:
    if str(asset.availability_status) != "available":
        raise DemoAssetResolutionError(
            f"Catalog asset {asset.asset_id} is not available (status={asset.availability_status})."
        )

    logical_path = asset.relative_path or asset.path
    resolved = _normalize_catalog_asset_path(logical_path)
    if not resolved.is_file():
        raise DemoAssetResolutionError(f"Demo asset file is missing for {asset.asset_id}.")
    return resolved


def _load_catalog_payload() -> CatalogPayloadModel:
    catalog_path = _catalog_path()
    if not catalog_path.is_file():
        raise DemoCatalogError(f"Demo catalog is missing: {catalog_path}")

    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoCatalogError(f"Demo catalog is not valid JSON: {catalog_path}") from exc
    except OSError as exc:
        raise DemoCatalogError(f"Demo catalog could not be read: {catalog_path}") from exc

    try:
        return CatalogPayloadModel.model_validate(payload)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        raise DemoCatalogError(f"Demo catalog failed validation: {message}") from exc


def _build_public_case(
    case: CatalogVerifyCaseModel,
    *,
    dataset_labels: Mapping[str, str],
) -> DemoCase:
    label, ground_truth_text = _ground_truth_payload(case.ground_truth)
    benchmark_context = normalize_benchmark_context(case.benchmark_context, split=case.split)
    canonical_recommended_method = benchmark_method_to_canonical(case.recommended_method)

    try:
        recommended_method = MatchMethod(canonical_recommended_method)
    except ValueError as exc:
        raise DemoCatalogError(
            f"Demo case {case.case_id} has unsupported recommended_method={case.recommended_method!r}."
        ) from exc

    benchmark_method = benchmark_context.get("benchmark_method")
    benchmark_run = benchmark_context.get("benchmark_run") or benchmark_context.get("run")
    benchmark_score = benchmark_context.get("benchmark_score")
    curation_rule = case.selection_reason or case.selection_policy
    if benchmark_score is not None:
        try:
            benchmark_score_value: Optional[float] = float(benchmark_score)
        except (TypeError, ValueError) as exc:
            raise DemoCatalogError(
                f"Demo case {case.case_id} has a non-numeric benchmark_score={benchmark_score!r}."
            ) from exc
    else:
        benchmark_score_value = None

    return DemoCase(
        id=case.case_id,
        title=case.title,
        description=case.description,
        difficulty=case.difficulty,
        dataset=case.dataset,
        dataset_label=dataset_labels.get(case.dataset, case.dataset),
        split=case.split,
        split_label=_split_label(case.split),
        label=label,
        ground_truth=ground_truth_text,
        recommended_method=recommended_method,
        benchmark_method=str(benchmark_method) if benchmark_method is not None else None,
        benchmark_run=str(benchmark_run) if benchmark_run is not None else None,
        curation_rule=curation_rule,
        benchmark_score=benchmark_score_value,
        capture_a=case.capture_a,
        capture_b=case.capture_b,
        image_a_url=_safe_asset_url(case.case_id, "a"),
        image_b_url=_safe_asset_url(case.case_id, "b"),
        asset_a_id=case.image_a.asset_id,
        asset_b_id=case.image_b.asset_id,
        case_type=case.case_type,
        availability_status=case.availability_status,
        selection_policy=case.selection_policy,
        tags=list(case.tags),
        modality_relation=case.modality_relation,
        evidence_quality=build_evidence_quality_summary(case.selection_diagnostics),
    )


def _build_demo_store_state() -> DemoStoreState:
    catalog = _load_catalog_payload()
    dataset_labels: Dict[str, str] = {}
    for raw_dataset in catalog.source_datasets:
        try:
            item = CatalogSourceDatasetModel.model_validate(raw_dataset)
        except ValidationError:
            continue
        dataset_labels[item.dataset] = item.dataset_label

    cases_by_id: Dict[str, ResolvedDemoRecord] = {}
    exclusions: Dict[str, str] = {}

    for index, raw_case in enumerate(catalog.verify_cases):
        case_key = f"<index:{index}>"
        if isinstance(raw_case, dict) and raw_case.get("case_id"):
            case_key = str(raw_case["case_id"])

        try:
            case = CatalogVerifyCaseModel.model_validate(raw_case)
        except ValidationError as exc:
            first_error = exc.errors()[0]["msg"] if exc.errors() else str(exc)
            exclusions[case_key] = f"case failed validation: {first_error}"
            continue

        reason: Optional[str] = None
        asset_paths: Dict[str, Path] = {}

        if _is_case_disabled(case):
            reason = "case is disabled or hidden"
        elif not bool(case.is_demo_safe):
            reason = "case is not marked demo-safe"
        elif str(case.availability_status) != "available":
            reason = f"case availability_status={case.availability_status!r}"
        else:
            try:
                asset_paths["a"] = _resolve_catalog_asset(case.image_a)
                asset_paths["b"] = _resolve_catalog_asset(case.image_b)
                public_case = _build_public_case(case, dataset_labels=dataset_labels)
            except DemoStoreError as exc:
                reason = str(exc)
            else:
                cases_by_id[case.case_id] = ResolvedDemoRecord(
                    public_case=public_case,
                    asset_paths=asset_paths,
                    sort_key=(
                        case.priority if case.priority is not None else 9999,
                        DIFFICULTY_ORDER.get(public_case.difficulty, 99),
                        public_case.dataset_label.lower(),
                        public_case.title.lower(),
                        public_case.id,
                    ),
                )

        if reason:
            exclusions[case.case_id] = reason

    if exclusions:
        logger.info(
            "Excluded %s demo case(s) while loading %s: %s",
            len(exclusions),
            _catalog_path(),
            exclusions,
        )

    catalog_build_health = build_catalog_build_health_summary(
        catalog.metadata.get("catalog_build_health"),
        case_evidence=[
            record.public_case.evidence_quality
            for record in cases_by_id.values()
            if record.public_case.evidence_quality is not None
        ],
    )

    return DemoStoreState(
        cases_by_id=cases_by_id,
        excluded_case_reasons=exclusions,
        catalog_build_health=catalog_build_health,
    )


@lru_cache(maxsize=1)
def _get_demo_store_state() -> DemoStoreState:
    return _build_demo_store_state()


def clear_demo_store_cache() -> None:
    _get_demo_store_state.cache_clear()


def get_demo_case_exclusions() -> Dict[str, str]:
    return dict(_get_demo_store_state().excluded_case_reasons)


def load_demo_cases() -> DemoCasesResponse:
    state = _get_demo_store_state()
    records = sorted(state.cases_by_id.values(), key=lambda record: record.sort_key)
    cases = [record.public_case for record in records]
    return DemoCasesResponse(cases=cases, catalog_build_health=state.catalog_build_health)


def resolve_demo_case_path(case_id: str, slot: str) -> Path:
    normalized_slot = str(slot).strip().lower()
    if normalized_slot not in {"a", "b"}:
        raise DemoInvalidSlotError(f"Unknown demo case slot: {slot!r}. Expected 'a' or 'b'.")

    state = _get_demo_store_state()
    record = state.cases_by_id.get(case_id)
    if record is None:
        if case_id in state.excluded_case_reasons:
            raise DemoCaseNotFoundError(
                f"Demo case {case_id!r} is not available: {state.excluded_case_reasons[case_id]}."
            )
        raise DemoCaseNotFoundError(f"Unknown demo case: {case_id!r}.")

    path = record.asset_paths[normalized_slot]
    if not path.is_file():
        raise DemoAssetResolutionError(f"Demo asset for case {case_id!r} and slot {normalized_slot!r} is missing.")
    if not _path_is_under(_assets_root(), path):
        raise DemoAssetResolutionError(
            f"Demo asset for case {case_id!r} and slot {normalized_slot!r} violates asset path policy."
        )
    return path
