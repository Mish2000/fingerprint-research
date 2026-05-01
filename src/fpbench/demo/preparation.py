from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

from fastapi import HTTPException
from fastapi.responses import FileResponse

import apps.api.main as api_main
from apps.api.catalog_store import (
    clear_catalog_store_cache,
    load_catalog_dataset_browser,
    load_catalog_identify_demo_identity_records,
    load_catalog_identify_probe_case_records,
    resolve_catalog_browser_asset_path,
)
from apps.api.demo_store import clear_demo_store_cache, get_demo_case_exclusions, load_demo_cases, resolve_demo_case_path
from ..catalog.demo_catalog import (
    IDENTITY_DATASET_ORDER,
    VERIFY_CASE_PLANS,
    build_browser_seed,
    build_catalog_bundle,
    build_identification_scenarios,
    build_identity_record,
    build_verify_case,
    load_benchmark_best,
    load_dataset_bundle,
)
from ..ui_assets.pipeline import DEFAULT_MAX_ITEMS_PER_DATASET, UiAssetConfig, build_ui_assets

ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = ROOT / "artifacts" / "reports" / "demo"
CATALOG_PATH = ROOT / "data" / "samples" / "catalog.json"
PREPARE_SCRIPT_PATH = ROOT / "scripts" / "prepare_demo_experience.py"

MODE_FULL = "full"
MODE_REBUILD_ONLY = "rebuild-assets-only"
MODE_SEED_ONLY = "seed-only"
MODE_HEALTH_ONLY = "health-check-only"
SUPPORTED_MODES = (MODE_FULL, MODE_REBUILD_ONLY, MODE_SEED_ONLY, MODE_HEALTH_ONLY)
VERIFY_METHODS = {
    "classic",
    "classic_v2",
    "classic_orb",
    "classic_gftt_orb",
    "harris",
    "sift",
    "dl",
    "dedicated",
    "vit",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_emit(emit: Optional[Callable[[str], None]]) -> Callable[[str], None]:
    if emit is not None:
        return emit
    return print


def _required_pair_splits() -> Dict[str, set[str]]:
    required: Dict[str, set[str]] = {}
    for plan in VERIFY_CASE_PLANS:
        required.setdefault(str(plan["dataset"]), set()).add(str(plan["split"]))
    return required


def _case_is_enabled_for_demo(case_payload: Dict[str, Any]) -> bool:
    if not bool(case_payload.get("is_demo_safe")):
        return False
    if bool(case_payload.get("hidden")):
        return False
    if bool(case_payload.get("disabled")):
        return False
    if case_payload.get("enabled") is False:
        return False
    return True


def _materialized_asset_path(asset_payload: Dict[str, Any], repo_root: Path) -> Optional[Path]:
    logical = str(asset_payload.get("relative_path") or asset_payload.get("path") or "").replace("\\", "/").strip()
    if not logical:
        return None
    return (repo_root / logical).resolve()


def _safe_file_response_path(response: FileResponse) -> Path:
    path = Path(str(response.path)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Served asset is missing on disk: {path}")
    return path


def _asset_source_relative_path(asset: Any) -> Optional[str]:
    value = getattr(asset, "source_relative_path", None)
    if value:
        return str(value)
    detail = getattr(asset, "availability_detail", None)
    if detail is not None:
        detail_value = getattr(detail, "source_relative_path", None)
        if detail_value:
            return str(detail_value)
    return None


def _asset_source_exists(asset: Any) -> bool:
    detail = getattr(asset, "availability_detail", None)
    if detail is not None:
        source_local_exists = getattr(detail, "source_local_exists", None)
        if source_local_exists is not None:
            return bool(source_local_exists)
    return False


def _collect_upstream_assets(
    *,
    verify_cases: Iterable[Any],
    identity_records: Iterable[Any],
    scenarios: Iterable[Any],
    browser_seed_entries: Iterable[Any],
) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for case in verify_cases:
        items.append((f"verify_case:{case.case_id}:image_a", case.image_a))
        items.append((f"verify_case:{case.case_id}:image_b", case.image_b))
    for identity in identity_records:
        for asset in identity.exemplars:
            items.append((f"identity:{identity.identity_id}:{asset.asset_id}", asset))
    for scenario in scenarios:
        items.append((f"scenario:{scenario.scenario_id}:probe_asset", scenario.probe_asset))
    for entry in browser_seed_entries:
        for asset in entry.items:
            items.append((f"browser_seed:{entry.dataset}:{asset.asset_id}", asset))
    return items


@dataclass
class PrepareOptions:
    mode: str = MODE_FULL
    repo_root: Path = ROOT
    report_root: Path = REPORT_ROOT
    max_items_per_dataset: int = DEFAULT_MAX_ITEMS_PER_DATASET
    emit: Optional[Callable[[str], None]] = None


@dataclass
class StepResult:
    name: str
    label: str
    status: str = "pending"
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def add_warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def add_failure(self, message: str) -> None:
        self.failures.append(str(message))

    def add_note(self, message: str) -> None:
        self.notes.append(str(message))

    def finalize(self) -> "StepResult":
        if self.failures:
            self.status = "failed"
        elif self.warnings:
            self.status = "passed_with_warnings"
        else:
            self.status = "passed"
        return self

    @classmethod
    def skipped(cls, name: str, label: str, reason: str) -> "StepResult":
        step = cls(name=name, label=label, status="skipped")
        step.add_note(reason)
        return step


@dataclass
class PrepareContext:
    options: PrepareOptions
    emit: Callable[[str], None]
    catalog_bundle: Optional[Dict[str, Any]] = None
    ui_assets_registry: Optional[Dict[str, Any]] = None
    api_started: bool = False


def _emit_step_outcome(ctx: PrepareContext, result: StepResult) -> None:
    badge = {
        "passed": "PASS",
        "passed_with_warnings": "WARN",
        "failed": "FAIL",
        "skipped": "SKIP",
    }.get(result.status, result.status.upper())
    ctx.emit(f"  {badge} {result.label} ({result.duration_ms:.0f} ms)")
    for message in result.failures[:3]:
        ctx.emit(f"    failure: {message}")
    for message in result.warnings[:3]:
        ctx.emit(f"    warning: {message}")
    for message in result.notes[:2]:
        ctx.emit(f"    note: {message}")


def _run_step(
    ctx: PrepareContext,
    *,
    index: int,
    total: int,
    name: str,
    label: str,
    fn: Callable[[PrepareContext], StepResult],
) -> StepResult:
    ctx.emit(f"[{index}/{total}] {label}")
    started = datetime.now(timezone.utc)
    try:
        result = fn(ctx)
    except Exception as exc:
        result = StepResult(name=name, label=label)
        result.add_failure(f"{type(exc).__name__}: {exc}")
        result.add_note("Unexpected exception while running the step.")
        result.add_note("".join(traceback.format_exception_only(type(exc), exc)).strip())
    result.duration_ms = max((datetime.now(timezone.utc) - started).total_seconds() * 1000.0, 0.0)
    if result.status == "pending":
        result.finalize()
    _emit_step_outcome(ctx, result)
    return result


def _ensure_api_started(ctx: PrepareContext, step: StepResult) -> bool:
    if ctx.api_started:
        return True
    try:
        api_main._startup()
    except Exception as exc:
        step.add_failure(f"API startup raised {type(exc).__name__}: {exc}")
        return False
    ctx.api_started = True
    return True


def _step_prerequisites(ctx: PrepareContext) -> StepResult:
    step = StepResult(name="prerequisites", label="Prerequisites")
    repo_root = ctx.options.repo_root
    required_dirs = [
        repo_root / "data",
        repo_root / "data" / "manifests",
        repo_root / "data" / "samples",
        repo_root / "data" / "processed",
        repo_root / "scripts",
        repo_root / "apps" / "api",
        repo_root / "src",
    ]
    required_files = [
        repo_root / "scripts" / "build_demo_catalog.py",
        repo_root / "scripts" / "build_ui_assets.py",
        repo_root / "apps" / "api" / "main.py",
        repo_root / "apps" / "api" / "identify_demo_store.py",
        repo_root / "apps" / "api" / "catalog_store.py",
    ]
    pair_splits = _required_pair_splits()
    for dataset in IDENTITY_DATASET_ORDER:
        dataset_root = repo_root / "data" / "manifests" / dataset
        required_dirs.append(dataset_root)
        required_files.extend(
            [
                dataset_root / "manifest.csv",
                dataset_root / "stats.json",
                dataset_root / "split.json",
            ]
        )
        for split in sorted(pair_splits.get(dataset, set())):
            required_files.append(dataset_root / f"pairs_{split}.csv")

    missing_dirs = [path for path in required_dirs if not path.is_dir()]
    missing_files = [path for path in required_files if not path.is_file()]
    for path in missing_dirs:
        step.add_failure(f"Missing required directory: {path}")
    for path in missing_files:
        step.add_failure(f"Missing required file: {path}")

    writable_dirs = [
        repo_root / "data" / "samples",
        repo_root / "data" / "processed",
    ]
    for path in writable_dirs:
        if path.exists() and not os.access(path, os.W_OK):
            step.add_failure(f"Output directory is not writable: {path}")

    try:
        ctx.options.report_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        step.add_failure(f"Could not create report directory {ctx.options.report_root}: {exc}")
    else:
        if not os.access(ctx.options.report_root, os.W_OK):
            step.add_failure(f"Report directory is not writable: {ctx.options.report_root}")

    step.metrics = {
        "required_directories_checked": len(required_dirs),
        "required_files_checked": len(required_files),
        "missing_directories": len(missing_dirs),
        "missing_files": len(missing_files),
    }
    return step.finalize()


def _step_raw_data(ctx: PrepareContext) -> StepResult:
    step = StepResult(name="raw_data", label="Raw Data Verification")
    benchmark_best = load_benchmark_best()
    bundles: Dict[str, Any] = {}
    manifest_rows_total = 0
    verify_cases: list[Any] = []
    identity_records: list[Any] = []
    browser_seed_entries: list[Any] = []

    for dataset in IDENTITY_DATASET_ORDER:
        try:
            bundle = load_dataset_bundle(dataset, benchmark_best)
        except Exception as exc:
            step.add_failure(f"Could not load source bundle for {dataset}: {exc}")
            continue
        bundles[dataset] = bundle
        manifest_rows = int(getattr(bundle.manifest, "shape", (0, 0))[0])
        manifest_rows_total += manifest_rows
        if manifest_rows <= 0:
            step.add_failure(f"Dataset {dataset} has an empty manifest.")

    if step.failures:
        step.metrics = {
            "datasets_checked": len(bundles),
            "manifest_rows_checked": manifest_rows_total,
        }
        return step.finalize()

    for plan in VERIFY_CASE_PLANS:
        dataset = str(plan["dataset"])
        try:
            case = build_verify_case(bundles[dataset], plan)
        except Exception as exc:
            step.add_failure(f"Failed to derive verify case for {dataset}/{plan['case_type']}: {exc}")
            continue
        if case is None:
            step.add_failure(
                f"Could not derive required verify case for {dataset}/{plan['split']}/{plan['case_type']}."
            )
            continue
        verify_cases.append(case)

    for dataset in IDENTITY_DATASET_ORDER:
        try:
            identity_records.append(build_identity_record(bundles[dataset]))
        except Exception as exc:
            step.add_failure(f"Failed to derive identity demo record for {dataset}: {exc}")
        try:
            browser_seed_entries.append(build_browser_seed(bundles[dataset]))
        except Exception as exc:
            step.add_failure(f"Failed to derive browser seed for {dataset}: {exc}")

    scenarios: list[Any] = []
    if not step.failures:
        try:
            scenarios = build_identification_scenarios(bundles, identity_records)
        except Exception as exc:
            step.add_failure(f"Failed to derive identification demo scenarios: {exc}")

    unique_sources: set[str] = set()
    upstream_assets = _collect_upstream_assets(
        verify_cases=verify_cases,
        identity_records=identity_records,
        scenarios=scenarios,
        browser_seed_entries=browser_seed_entries,
    )
    for location, asset in upstream_assets:
        relative_source = _asset_source_relative_path(asset)
        if not relative_source:
            step.add_failure(f"Asset {location} is not traceable to a repo-local source path.")
            continue
        unique_sources.add(relative_source)
        if not relative_source.replace("\\", "/").startswith("data/"):
            step.add_failure(f"Asset {location} points outside the project data roots: {relative_source}")
            continue
        if not _asset_source_exists(asset):
            step.add_failure(f"Required raw asset is missing for {location}: {relative_source}")

    step.metrics = {
        "datasets_checked": len(bundles),
        "manifest_rows_checked": manifest_rows_total,
        "verify_cases_derived": len(verify_cases),
        "identity_records_derived": len(identity_records),
        "browser_seed_entries_derived": len(browser_seed_entries),
        "identification_scenarios_derived": len(scenarios),
        "required_upstream_assets_checked": len(unique_sources),
    }
    return step.finalize()


def _step_catalog(ctx: PrepareContext) -> StepResult:
    step = StepResult(name="catalog", label="Catalog Build / Refresh")
    try:
        bundle = build_catalog_bundle(write_files=True)
    except Exception as exc:
        step.add_failure(f"Catalog build failed: {exc}")
        return step.finalize()

    ctx.catalog_bundle = bundle
    report = bundle["report"]
    catalog = bundle["catalog"]
    if report.get("validation_status") != "pass":
        for message in report.get("errors", [])[:10]:
            step.add_failure(message)
    for message in report.get("warnings", [])[:10]:
        step.add_warning(message)

    step.metrics = {
        "verify_cases": int(catalog.get("metadata", {}).get("total_verify_cases", len(catalog.get("verify_cases", [])))),
        "identity_records": int(
            catalog.get("metadata", {}).get(
                "total_identity_records",
                len(catalog.get("identify_gallery", {}).get("identities", [])),
            )
        ),
        "browser_seed_items": int(catalog.get("metadata", {}).get("total_browser_seed_items", 0)),
        "materialized_asset_count": int(catalog.get("metadata", {}).get("materialized_asset_count", 0)),
        "validation_errors": int(report.get("validation_errors_count", 0)),
        "validation_warnings": int(report.get("validation_warnings_count", 0)),
    }
    return step.finalize()


def _catalog_datasets_for_build(ctx: PrepareContext) -> list[str]:
    if ctx.catalog_bundle:
        payload = ctx.catalog_bundle.get("catalog", {})
        source_datasets = payload.get("source_datasets", [])
        datasets = [str(item.get("dataset")) for item in source_datasets if item.get("dataset")]
        if datasets:
            return datasets
        included = payload.get("metadata", {}).get("included_datasets", [])
        if included:
            return [str(item) for item in included]
    return list(IDENTITY_DATASET_ORDER)


def _browser_preview_exclusions(ctx: PrepareContext) -> Dict[str, str]:
    exclusions: Dict[str, str] = {}
    registry = ctx.ui_assets_registry or {}
    for item in registry.get("datasets", []):
        dataset = str(item.get("dataset") or "").strip()
        if not dataset:
            continue
        report_rel = str(item.get("validation_report_path") or "").replace("\\", "/").strip()
        if not report_rel:
            continue
        report_path = (ctx.options.repo_root / report_rel).resolve()
        if not report_path.is_file():
            continue
        try:
            report_payload = _load_json(report_path)
        except Exception:
            continue
        if report_payload.get("browser_preview_enabled") is False:
            reason = str(report_payload.get("browser_preview_exclusion_reason") or "").strip()
            exclusions[dataset] = reason or "browser previews intentionally disabled"
    return exclusions


def _step_ui_assets(ctx: PrepareContext) -> StepResult:
    step = StepResult(name="ui_assets", label="Thumbnails / Preview Assets")
    datasets = _catalog_datasets_for_build(ctx)
    try:
        registry = build_ui_assets(
            datasets,
            repo_root=ctx.options.repo_root,
            config=UiAssetConfig(max_items_per_dataset=ctx.options.max_items_per_dataset),
        )
    except Exception as exc:
        step.add_failure(f"UI asset build failed: {exc}")
        return step.finalize()

    ctx.ui_assets_registry = registry
    dataset_results = {str(item["dataset"]): item for item in registry.get("datasets", [])}
    assets_built_total = 0
    browser_preview_excluded = 0
    for dataset in datasets:
        item = dataset_results.get(dataset)
        if item is None:
            step.add_failure(f"UI assets registry is missing dataset {dataset}.")
            continue
        assets_built_total += int(item.get("item_count", 0))
        status = str(item.get("validation_status", "unknown"))
        report_payload: Dict[str, Any] = {}
        report_rel = str(item.get("validation_report_path") or "").replace("\\", "/").strip()
        if report_rel:
            report_path = (ctx.options.repo_root / report_rel).resolve()
            if report_path.is_file():
                try:
                    report_payload = _load_json(report_path)
                except Exception as exc:
                    step.add_warning(f"Could not parse UI assets validation report for {dataset}: {exc}")

        browser_preview_enabled = report_payload.get("browser_preview_enabled")
        exclusion_reason = str(report_payload.get("browser_preview_exclusion_reason") or "").strip()
        if browser_preview_enabled is False:
            browser_preview_excluded += 1
            note = exclusion_reason or "dataset is intentionally excluded from browser previews"
            step.add_note(f"Browser preview exclusion respected for {dataset}: {note}.")
            continue

        if int(item.get("item_count", 0)) <= 0:
            step.add_failure(f"No preview assets were built for demo dataset {dataset}.")
        if status == "fail":
            step.add_failure(f"UI assets validation failed for {dataset}.")
        elif status != "pass":
            step.add_note(f"UI assets for {dataset} completed with status={status}.")

    step.metrics = {
        "datasets_built": len(dataset_results),
        "required_datasets": len(datasets),
        "assets_built": assets_built_total,
        "browser_preview_excluded": browser_preview_excluded,
    }
    return step.finalize()


def _step_sanity(ctx: PrepareContext) -> StepResult:
    step = StepResult(name="sanity", label="Demo Case Sanity Checks")
    repo_root = ctx.options.repo_root
    browser_exclusions = _browser_preview_exclusions(ctx)
    catalog_path = repo_root / "data" / "samples" / "catalog.json"
    if not catalog_path.is_file():
        step.add_failure(f"Catalog is missing: {catalog_path}")
        return step.finalize()

    payload = _load_json(catalog_path)
    verify_cases = payload.get("verify_cases", [])
    identify_gallery = payload.get("identify_gallery", {})
    dataset_browser_seed = payload.get("dataset_browser_seed", [])

    case_ids = [str(item.get("case_id")) for item in verify_cases if item.get("case_id")]
    if len(case_ids) != len(set(case_ids)):
        step.add_failure("Catalog verify case IDs are not unique.")

    for case in verify_cases:
        case_id = str(case.get("case_id") or "<missing-case-id>")
        if not str(case.get("recommended_method") or "").strip():
            step.add_failure(f"Verify case {case_id} is missing recommended_method.")
        elif str(case.get("recommended_method")).strip() not in VERIFY_METHODS:
            step.add_failure(
                f"Verify case {case_id} has unsupported recommended_method={case.get('recommended_method')!r}."
            )
        if not str(case.get("difficulty") or "").strip():
            step.add_failure(f"Verify case {case_id} is missing difficulty.")
        if not str(case.get("ground_truth") or "").strip():
            step.add_failure(f"Verify case {case_id} is missing ground_truth.")
        for slot_name in ("image_a", "image_b"):
            asset_payload = case.get(slot_name) or {}
            asset_path = _materialized_asset_path(asset_payload, repo_root)
            if asset_path is None:
                step.add_failure(f"Verify case {case_id} has no materialized path for {slot_name}.")
                continue
            if not asset_path.is_file():
                step.add_failure(f"Verify case {case_id} references missing asset {asset_path}.")
            try:
                asset_path.relative_to((repo_root / "data" / "samples" / "assets").resolve())
            except Exception:
                step.add_failure(f"Verify case {case_id} asset escapes data/samples/assets: {asset_path}")

    clear_catalog_store_cache()
    clear_demo_store_cache()
    demo_safe_case_ids = {
        str(case["case_id"])
        for case in verify_cases
        if case.get("case_id") and _case_is_enabled_for_demo(case)
    }
    demo_cases_response = load_demo_cases()
    loaded_demo_case_ids = {case.id for case in demo_cases_response.cases}
    missing_demo_cases = sorted(demo_safe_case_ids - loaded_demo_case_ids)
    exclusions = get_demo_case_exclusions()
    for case_id in missing_demo_cases:
        reason = exclusions.get(case_id, "case did not survive demo-case loading")
        step.add_failure(f"Demo case {case_id} is unavailable for the demo flow: {reason}")
    if not demo_cases_response.cases:
        step.add_failure("No verify demo cases are currently loadable from the demo store.")
    for case in demo_cases_response.cases:
        try:
            resolve_demo_case_path(case.id, "a")
            resolve_demo_case_path(case.id, "b")
        except Exception as exc:
            step.add_failure(f"Demo case {case.id} cannot serve its assets: {exc}")

    demo_safe_identity_ids = {
        str(identity["identity_id"])
        for identity in identify_gallery.get("identities", [])
        if identity.get("identity_id") and bool(identity.get("is_demo_safe"))
    }
    demo_identity_records = load_catalog_identify_demo_identity_records()
    loaded_demo_identity_ids = {record.public_item.id for record in demo_identity_records}
    missing_identities = sorted(demo_safe_identity_ids - loaded_demo_identity_ids)
    for identity_id in missing_identities:
        step.add_failure(f"Demo identity {identity_id} is not seedable from the catalog/browser layers.")
    if not demo_identity_records:
        step.add_failure("No demo-safe identification identities are currently available.")
    for record in demo_identity_records:
        if not record.enrollment_asset_path.is_file():
            step.add_failure(
                f"Enrollment preview asset is missing for demo identity {record.public_item.id}: {record.enrollment_asset_path}"
            )

    probe_case_records = load_catalog_identify_probe_case_records()
    if not probe_case_records:
        step.add_failure("No identification probe presets are currently available for the demo flow.")
    for record in probe_case_records:
        if not record.probe_asset_path.is_file():
            step.add_failure(f"Probe preview asset is missing for identify probe {record.public_item.id}.")

    browser_seed_datasets = [str(entry.get("dataset")) for entry in dataset_browser_seed if entry.get("dataset")]
    browser_datasets_skipped = 0
    for dataset in browser_seed_datasets:
        if dataset in browser_exclusions:
            browser_datasets_skipped += 1
            step.add_note(f"Skipping dataset browser sanity for {dataset}: {browser_exclusions[dataset]}.")
            continue
        try:
            browser_response = load_catalog_dataset_browser(dataset=dataset, limit=1, offset=0)
        except Exception as exc:
            step.add_failure(f"Dataset browser is unavailable for {dataset}: {exc}")
            continue
        if browser_response.total <= 0 or not browser_response.items:
            step.add_failure(f"Dataset browser has no loadable preview items for {dataset}.")
            continue
        first_item = browser_response.items[0]
        try:
            resolve_catalog_browser_asset_path(dataset, first_item.asset_id, "thumbnail")
            resolve_catalog_browser_asset_path(dataset, first_item.asset_id, "preview")
        except Exception as exc:
            step.add_failure(f"Dataset browser asset serving failed for {dataset}/{first_item.asset_id}: {exc}")

    if exclusions:
        non_demo_exclusions = [
            f"{case_id} ({reason})"
            for case_id, reason in sorted(exclusions.items())
            if case_id not in demo_safe_case_ids
        ]
        if non_demo_exclusions:
            step.add_note(
                f"Excluded {len(non_demo_exclusions)} non-demo verify case(s) from /demo/cases as expected."
            )

    step.metrics = {
        "verify_cases_checked": len(verify_cases),
        "demo_cases_available": len(demo_cases_response.cases),
        "demo_safe_verify_cases_expected": len(demo_safe_case_ids),
        "demo_identities_available": len(demo_identity_records),
        "demo_safe_identities_expected": len(demo_safe_identity_ids),
        "probe_cases_available": len(probe_case_records),
        "browser_datasets_checked": len(browser_seed_datasets),
        "browser_datasets_skipped": browser_datasets_skipped,
    }
    return step.finalize()


def _step_seed(ctx: PrepareContext) -> StepResult:
    step = StepResult(name="seed", label="Identification Demo Seed")
    if not _ensure_api_started(ctx, step):
        return step.finalize()

    try:
        reset_response = api_main.identify_demo_reset()
    except HTTPException as exc:
        step.add_failure(f"Identify demo reset failed ({exc.status_code}): {exc.detail}")
        return step.finalize()
    except Exception as exc:
        step.add_failure(f"Identify demo reset failed: {exc}")
        return step.finalize()

    try:
        seed_response = api_main.identify_demo_seed()
    except HTTPException as exc:
        step.add_failure(f"Identify demo seed failed ({exc.status_code}): {exc.detail}")
        return step.finalize()
    except Exception as exc:
        step.add_failure(f"Identify demo seed failed: {exc}")
        return step.finalize()

    try:
        stats_response = api_main.identify_stats()
    except HTTPException as exc:
        step.add_failure(f"Identify demo stats failed after seed ({exc.status_code}): {exc.detail}")
        return step.finalize()
    except Exception as exc:
        step.add_failure(f"Identify demo stats failed after seed: {exc}")
        return step.finalize()

    if int(seed_response.demo_seeded_count) <= 0:
        step.add_failure("Identification demo store is still empty after seeding.")
    if int(stats_response.demo_seeded_count) != int(seed_response.demo_seeded_count):
        step.add_failure(
            "Identification demo seeded count does not match post-seed stats "
            f"({seed_response.demo_seeded_count} vs {stats_response.demo_seeded_count})."
        )
    extra_identities = int(seed_response.total_enrolled) - int(seed_response.demo_seeded_count)
    if extra_identities > 0:
        step.add_warning(
            f"Identification store still contains {extra_identities} non-demo identities after reset+seed."
        )

    step.metrics = {
        "removed_before_seed": int(reset_response.removed_count),
        "seeded_count": int(seed_response.seeded_count),
        "updated_count": int(seed_response.updated_count),
        "total_enrolled": int(seed_response.total_enrolled),
        "demo_seeded_count": int(seed_response.demo_seeded_count),
    }
    return step.finalize()


def _step_api_health(ctx: PrepareContext) -> StepResult:
    step = StepResult(name="api_health", label="API Health / Integration Checks")
    if not _ensure_api_started(ctx, step):
        return step.finalize()

    endpoints_checked = 0
    payload = api_main.health()
    endpoints_checked += 1
    if not bool(payload.get("ok")):
        step.add_failure(f"API match service is not healthy: {payload.get('error')}")
    if not bool(payload.get("identify_ok")):
        step.add_failure(f"API identification service is not healthy: {payload.get('identify_error')}")

    try:
        datasets_response = api_main.catalog_datasets()
        endpoints_checked += 1
    except HTTPException as exc:
        step.add_failure(f"/catalog/datasets failed ({exc.status_code}): {exc.detail}")
        datasets_response = None
    except Exception as exc:
        step.add_failure(f"/catalog/datasets failed: {exc}")
        datasets_response = None

    try:
        verify_response = api_main.catalog_verify_cases(limit=5, offset=0)
        endpoints_checked += 1
    except HTTPException as exc:
        step.add_failure(f"/catalog/verify-cases failed ({exc.status_code}): {exc.detail}")
        verify_response = None
    except Exception as exc:
        step.add_failure(f"/catalog/verify-cases failed: {exc}")
        verify_response = None
    else:
        if verify_response.total <= 0:
            step.add_failure("/catalog/verify-cases returned no items.")

    try:
        demo_cases_response = api_main.demo_cases()
        endpoints_checked += 1
    except HTTPException as exc:
        step.add_failure(f"/demo/cases failed ({exc.status_code}): {exc.detail}")
        demo_cases_response = None
    except Exception as exc:
        step.add_failure(f"/demo/cases failed: {exc}")
        demo_cases_response = None
    else:
        if not demo_cases_response.cases:
            step.add_failure("/demo/cases returned no demo-ready verify cases.")
        else:
            first_case = demo_cases_response.cases[0]
            try:
                response_a = api_main.demo_case_asset(first_case.id, "a")
                response_b = api_main.demo_case_asset(first_case.id, "b")
                _safe_file_response_path(response_a)
                _safe_file_response_path(response_b)
                endpoints_checked += 2
            except HTTPException as exc:
                step.add_failure(f"Demo asset serving failed for case {first_case.id} ({exc.status_code}): {exc.detail}")
            except Exception as exc:
                step.add_failure(f"Demo asset serving failed for case {first_case.id}: {exc}")

    try:
        identify_gallery_response = api_main.catalog_identify_gallery(limit=5, offset=0)
        endpoints_checked += 1
    except HTTPException as exc:
        step.add_failure(f"/catalog/identify-gallery failed ({exc.status_code}): {exc.detail}")
        identify_gallery_response = None
    except Exception as exc:
        step.add_failure(f"/catalog/identify-gallery failed: {exc}")
        identify_gallery_response = None
    else:
        if not identify_gallery_response.demo_identities:
            step.add_failure("/catalog/identify-gallery returned no demo identities.")
        if not identify_gallery_response.probe_cases:
            step.add_failure("/catalog/identify-gallery returned no probe presets.")

    browser_asset_checks = 0
    if datasets_response is not None:
        browser_datasets = [item for item in datasets_response.items if bool(item.has_browser_assets)]
        if not browser_datasets:
            step.add_failure("No dataset exposes browser assets through the catalog API.")
        else:
            dataset = browser_datasets[0].dataset
            try:
                browser_response = api_main.catalog_dataset_browser(dataset=dataset, limit=1, offset=0)
                endpoints_checked += 1
            except HTTPException as exc:
                step.add_failure(f"/catalog/dataset-browser failed for {dataset} ({exc.status_code}): {exc.detail}")
            except Exception as exc:
                step.add_failure(f"/catalog/dataset-browser failed for {dataset}: {exc}")
            else:
                if browser_response.total <= 0 or not browser_response.items:
                    step.add_failure(f"/catalog/dataset-browser returned no items for {dataset}.")
                else:
                    first_item = browser_response.items[0]
                    try:
                        thumb_response = api_main.catalog_asset(dataset, first_item.asset_id, "thumbnail")
                        preview_response = api_main.catalog_asset(dataset, first_item.asset_id, "preview")
                        _safe_file_response_path(thumb_response)
                        _safe_file_response_path(preview_response)
                        browser_asset_checks = 2
                        endpoints_checked += 2
                    except HTTPException as exc:
                        step.add_failure(
                            f"Catalog asset serving failed for {dataset}/{first_item.asset_id} "
                            f"({exc.status_code}): {exc.detail}"
                        )
                    except Exception as exc:
                        step.add_failure(f"Catalog asset serving failed for {dataset}/{first_item.asset_id}: {exc}")

    try:
        stats_response = api_main.identify_stats()
        endpoints_checked += 1
    except HTTPException as exc:
        step.add_failure(f"/identify/stats failed ({exc.status_code}): {exc.detail}")
        stats_response = None
    except Exception as exc:
        step.add_failure(f"/identify/stats failed: {exc}")
        stats_response = None
    else:
        if int(stats_response.demo_seeded_count) <= 0:
            step.add_failure("Identification demo stats show zero demo-seeded identities.")

    step.metrics = {
        "datasets_exposed": len(datasets_response.items) if datasets_response is not None else 0,
        "verify_cases_total": int(verify_response.total) if verify_response is not None else 0,
        "demo_cases_total": len(demo_cases_response.cases) if demo_cases_response is not None else 0,
        "demo_identities_total": len(identify_gallery_response.demo_identities) if identify_gallery_response is not None else 0,
        "probe_cases_total": len(identify_gallery_response.probe_cases) if identify_gallery_response is not None else 0,
        "browser_asset_checks": browser_asset_checks,
        "demo_seeded_count": int(stats_response.demo_seeded_count) if stats_response is not None else 0,
        "endpoints_checked": endpoints_checked,
    }
    return step.finalize()


def _selected_steps(mode: str) -> list[tuple[str, str, Callable[[PrepareContext], StepResult]]]:
    if mode == MODE_FULL:
        return [
            ("prerequisites", "Prerequisites", _step_prerequisites),
            ("raw_data", "Raw Data Verification", _step_raw_data),
            ("catalog", "Catalog Build / Refresh", _step_catalog),
            ("ui_assets", "Thumbnails / Preview Assets", _step_ui_assets),
            ("sanity", "Demo Case Sanity Checks", _step_sanity),
            ("seed", "Identification Demo Seed", _step_seed),
            ("api_health", "API Health / Integration Checks", _step_api_health),
        ]
    if mode == MODE_REBUILD_ONLY:
        return [
            ("prerequisites", "Prerequisites", _step_prerequisites),
            ("raw_data", "Raw Data Verification", _step_raw_data),
            ("catalog", "Catalog Build / Refresh", _step_catalog),
            ("ui_assets", "Thumbnails / Preview Assets", _step_ui_assets),
            ("sanity", "Demo Case Sanity Checks", _step_sanity),
        ]
    if mode == MODE_SEED_ONLY:
        return [
            ("prerequisites", "Prerequisites", _step_prerequisites),
            ("sanity", "Demo Case Sanity Checks", _step_sanity),
            ("seed", "Identification Demo Seed", _step_seed),
        ]
    if mode == MODE_HEALTH_ONLY:
        return [
            ("prerequisites", "Prerequisites", _step_prerequisites),
            ("sanity", "Demo Case Sanity Checks", _step_sanity),
            ("api_health", "API Health / Integration Checks", _step_api_health),
        ]
    raise ValueError(f"Unsupported mode: {mode}")


def _overall_status(steps: Sequence[StepResult]) -> str:
    if any(step.status == "failed" for step in steps):
        return "failed"
    if any(step.status == "passed_with_warnings" for step in steps):
        return "partial"
    return "success"


def _verdict(mode: str, steps: Sequence[StepResult]) -> tuple[str, str]:
    if any(step.status == "failed" for step in steps):
        return "not ready", "At least one blocking preparation step failed."
    if mode in {MODE_REBUILD_ONLY, MODE_SEED_ONLY}:
        return "not ready", f"Mode '{mode}' does not run the full demo-readiness evaluation."
    return "demo ready", "All readiness-critical checks in the selected mode passed."


def _build_summary(
    ctx: PrepareContext,
    *,
    started_at: str,
    steps: Sequence[StepResult],
    report_path: Optional[Path],
) -> Dict[str, Any]:
    status = _overall_status(steps)
    verdict, verdict_reason = _verdict(ctx.options.mode, steps)
    step_map = {step.name: step for step in steps}
    raw_metrics = step_map.get("raw_data", StepResult("raw_data", "Raw Data Verification")).metrics
    sanity_metrics = step_map.get("sanity", StepResult("sanity", "Demo Case Sanity Checks")).metrics
    ui_metrics = step_map.get("ui_assets", StepResult("ui_assets", "Thumbnails / Preview Assets")).metrics
    health_metrics = step_map.get("api_health", StepResult("api_health", "API Health / Integration Checks")).metrics
    seed_metrics = step_map.get("seed", StepResult("seed", "Identification Demo Seed")).metrics

    warning_count = sum(len(step.warnings) for step in steps)
    failure_count = sum(len(step.failures) for step in steps)
    return {
        "script": str(PREPARE_SCRIPT_PATH),
        "mode": ctx.options.mode,
        "started_at": started_at,
        "finished_at": _now_utc(),
        "status": status,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "report_path": str(report_path) if report_path else None,
        "steps": [asdict(step) for step in steps],
        "totals": {
            "datasets_checked": int(raw_metrics.get("datasets_checked", 0)),
            "cases_checked": int(sanity_metrics.get("verify_cases_checked", 0)),
            "assets_built": int(ui_metrics.get("assets_built", 0)),
            "checks_passed": sum(1 for step in steps if step.status in {"passed", "passed_with_warnings"}),
            "warnings": warning_count,
            "failures": failure_count,
            "identification_demo_store_ready": int(
                seed_metrics.get("demo_seeded_count", health_metrics.get("demo_seeded_count", 0))
            )
            > 0,
            "api_health_passed": step_map.get("api_health", StepResult("api_health", "API Health / Integration Checks")).status
            in {"passed", "passed_with_warnings"},
        },
    }


def _write_summary_artifacts(report_root: Path, summary: Dict[str, Any]) -> Optional[Path]:
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timestamped = report_root / f"prepare_demo_experience_{stamp}.json"
    latest = report_root / "prepare_demo_experience.latest.json"
    summary["report_path"] = str(timestamped)
    _json_dump(timestamped, summary)
    _json_dump(latest, summary)
    return timestamped


def _emit_final_summary(ctx: PrepareContext, summary: Dict[str, Any]) -> None:
    ctx.emit("")
    ctx.emit("Summary")
    ctx.emit(f"  status: {summary['status']}")
    ctx.emit(f"  verdict: {summary['verdict']}")
    ctx.emit(f"  reason: {summary['verdict_reason']}")
    ctx.emit(
        "  totals: "
        f"datasets={summary['totals']['datasets_checked']}, "
        f"cases={summary['totals']['cases_checked']}, "
        f"assets_built={summary['totals']['assets_built']}, "
        f"checks_passed={summary['totals']['checks_passed']}, "
        f"warnings={summary['totals']['warnings']}, "
        f"failures={summary['totals']['failures']}"
    )
    ctx.emit(
        "  readiness: "
        f"identify_store_ready={summary['totals']['identification_demo_store_ready']}, "
        f"api_health_passed={summary['totals']['api_health_passed']}"
    )
    if summary.get("report_path"):
        ctx.emit(f"  report: {summary['report_path']}")


def run_prepare_demo_experience(options: PrepareOptions | None = None) -> Dict[str, Any]:
    options = options or PrepareOptions()
    if options.mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode {options.mode!r}. Expected one of {SUPPORTED_MODES}.")

    ctx = PrepareContext(options=options, emit=_coerce_emit(options.emit))
    started_at = _now_utc()
    steps: list[StepResult] = []
    selected_steps = _selected_steps(options.mode)
    total = len(selected_steps)
    halted = False

    for index, (name, label, fn) in enumerate(selected_steps, start=1):
        if halted:
            steps.append(StepResult.skipped(name, label, "Skipped because an earlier blocking step failed."))
            continue
        result = _run_step(ctx, index=index, total=total, name=name, label=label, fn=fn)
        steps.append(result)
        if result.status == "failed":
            halted = True

    summary = _build_summary(ctx, started_at=started_at, steps=steps, report_path=None)
    report_path = None
    try:
        report_path = _write_summary_artifacts(options.report_root, summary)
    except Exception as exc:
        summary["artifact_write_error"] = f"{type(exc).__name__}: {exc}"
    summary["report_path"] = str(report_path) if report_path else summary.get("report_path")
    _emit_final_summary(ctx, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the full fingerprint demo experience from one deterministic entry point."
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default=MODE_FULL,
        help="Preparation mode. 'full' is the demo-day default.",
    )
    parser.add_argument(
        "--max-items-per-dataset",
        type=int,
        default=DEFAULT_MAX_ITEMS_PER_DATASET,
        help="Maximum number of browser preview assets to build per dataset in modes that rebuild UI assets.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_prepare_demo_experience(
        PrepareOptions(
            mode=args.mode,
            max_items_per_dataset=args.max_items_per_dataset,
        )
    )
    return 0 if summary["verdict"] == "demo ready" else 1
