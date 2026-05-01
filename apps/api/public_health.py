from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from apps.api.schemas import CatalogBuildHealthSummary, CatalogDatasetDemoHealth, EvidenceQualitySummary


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n", ""}:
            return False
    return False


def _normalize_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(count, 0)


def _mapping_from(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence_of_mappings(values: Any) -> list[Mapping[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    return [item for item in values if isinstance(item, Mapping)]


def _dataset_key_only(value: Any) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    return text.split(":", 1)[0].strip() or None


def _public_selection_driver(raw_driver: str | None, *, benchmark_backed: bool, heuristic_used: bool) -> str:
    if benchmark_backed or raw_driver == "benchmark_driven":
        return "benchmark_driven"
    if heuristic_used or raw_driver in {"heuristic_fallback", "heuristic_only"}:
        return "heuristic_fallback"
    return "benchmark_driven"


def _evidence_status(
    raw_driver: str | None,
    *,
    benchmark_backed: bool,
    heuristic_used: bool,
) -> str:
    if benchmark_backed:
        return "strong"
    if heuristic_used:
        return "degraded"
    if raw_driver == "heuristic_only":
        return "fallback"
    return "fallback"


def _evidence_note(
    diagnostics: Mapping[str, Any],
    *,
    benchmark_backed: bool,
    heuristic_used: bool,
    evidence_status: str,
) -> str:
    fallback_category = _normalize_text(diagnostics.get("fallback_category"))
    discovery_outcome = _normalize_text(diagnostics.get("benchmark_discovery_outcome"))
    selection_status = _normalize_text(diagnostics.get("benchmark_selection_status"))

    if evidence_status == "strong" and benchmark_backed:
        return "Selected directly from benchmark evidence."

    if heuristic_used:
        if fallback_category in {
            "benchmark_resolution_missing",
            "score_file_missing",
            "score_file_unparsable",
            "score_file_missing_columns",
            "score_file_no_pair_overlap",
        }:
            return "Benchmark evidence was incomplete, so this case used a heuristic fallback."
        if discovery_outcome in {
            "dataset_fallback_no_benchmark_evidence",
            "benchmark_best_unavailable",
            "benchmark_best_missing",
        }:
            return "Benchmark evidence was unavailable, so this case used a heuristic fallback."
        return "This case used a heuristic fallback instead of full benchmark evidence."

    if selection_status == "not_requested" or _normalize_text(diagnostics.get("selection_driver")) == "heuristic_only":
        return "Selected from curated heuristics rather than benchmark evidence."

    return "Benchmark evidence is limited for this case."


def build_evidence_quality_summary(selection_diagnostics: Any) -> EvidenceQualitySummary | None:
    diagnostics = _mapping_from(selection_diagnostics)
    if not diagnostics:
        return None

    raw_driver = _normalize_text(diagnostics.get("selection_driver"))
    benchmark_backed = _normalize_bool(diagnostics.get("benchmark_backed_selection"))
    heuristic_used = _normalize_bool(diagnostics.get("heuristic_fallback_used"))
    discovery_outcome = _normalize_text(diagnostics.get("benchmark_discovery_outcome")) or "unknown"
    evidence_status = _evidence_status(
        raw_driver,
        benchmark_backed=benchmark_backed,
        heuristic_used=heuristic_used,
    )
    return EvidenceQualitySummary(
        selection_driver=_public_selection_driver(
            raw_driver,
            benchmark_backed=benchmark_backed,
            heuristic_used=heuristic_used,
        ),
        benchmark_backed_selection=benchmark_backed,
        heuristic_fallback_used=heuristic_used,
        benchmark_discovery_outcome=discovery_outcome,
        evidence_status=evidence_status,
        evidence_note=_evidence_note(
            diagnostics,
            benchmark_backed=benchmark_backed,
            heuristic_used=heuristic_used,
            evidence_status=evidence_status,
        ),
    )


def _dataset_health_note(
    *,
    status: str,
    planned_verify_cases: int,
    built_verify_cases: int,
    benchmark_backed_cases: int,
    heuristic_fallback_cases: int,
    missing_benchmark_evidence: bool,
) -> str:
    if status == "healthy":
        return f"{benchmark_backed_cases} curated verify case(s) are benchmark-backed and demo-ready."

    if status == "incomplete":
        if heuristic_fallback_cases > 0:
            return (
                f"Built {built_verify_cases} of {planned_verify_cases} planned verify case(s); "
                f"{heuristic_fallback_cases} currently rely on heuristic fallback."
            )
        return f"Built {built_verify_cases} of {planned_verify_cases} planned verify case(s)."

    if heuristic_fallback_cases > 0 and missing_benchmark_evidence:
        return (
            f"{heuristic_fallback_cases} of {built_verify_cases} curated verify case(s) "
            "use heuristic fallback because benchmark evidence is incomplete."
        )

    if heuristic_fallback_cases > 0:
        return f"{heuristic_fallback_cases} of {built_verify_cases} curated verify case(s) use heuristic fallback."

    return "Benchmark evidence is incomplete for this dataset's curated verify cases."


def build_dataset_demo_health_summary(
    verify_selection_diagnostics: Any,
    *,
    case_evidence: Sequence[EvidenceQualitySummary] = (),
    built_case_count: int = 0,
) -> CatalogDatasetDemoHealth | None:
    entries = _sequence_of_mappings(verify_selection_diagnostics)
    evidence_items = list(case_evidence)

    if not entries and not evidence_items and built_case_count <= 0:
        return None

    planned_verify_cases = sum(
        _normalize_count(entry.get("planned_curated_cases"))
        or len(entry.get("planned_case_ids") or [])
        for entry in entries
    )
    built_verify_cases = sum(
        _normalize_count(entry.get("curated_cases_built_successfully"))
        or len(entry.get("built_case_ids") or [])
        for entry in entries
    )

    if built_verify_cases == 0:
        built_verify_cases = max(built_case_count, len(evidence_items))
    if planned_verify_cases == 0:
        planned_verify_cases = max(built_verify_cases, len(evidence_items))

    benchmark_backed_cases = sum(
        _normalize_count(_mapping_from(entry.get("selection_driver_counts")).get("benchmark_driven"))
        for entry in entries
    )
    heuristic_fallback_cases = sum(
        _normalize_count(_mapping_from(entry.get("selection_driver_counts")).get("heuristic_fallback"))
        + _normalize_count(_mapping_from(entry.get("selection_driver_counts")).get("heuristic_only"))
        for entry in entries
    )

    if benchmark_backed_cases == 0 and heuristic_fallback_cases == 0 and evidence_items:
        benchmark_backed_cases = sum(1 for item in evidence_items if item.benchmark_backed_selection)
        heuristic_fallback_cases = sum(1 for item in evidence_items if item.selection_driver == "heuristic_fallback")

    missing_benchmark_evidence = any(
        (
            not _normalize_bool(entry.get("benchmark_best_available"))
            or not _normalize_bool(entry.get("benchmark_resolution_complete"))
            or _normalize_bool(entry.get("heuristic_fallback_used"))
        )
        for entry in entries
    )
    if not entries:
        missing_benchmark_evidence = heuristic_fallback_cases > 0

    if planned_verify_cases > built_verify_cases:
        status = "incomplete"
    elif heuristic_fallback_cases > 0 or missing_benchmark_evidence:
        status = "degraded"
    else:
        status = "healthy"

    return CatalogDatasetDemoHealth(
        planned_verify_cases=planned_verify_cases,
        built_verify_cases=built_verify_cases,
        benchmark_backed_cases=max(benchmark_backed_cases, 0),
        heuristic_fallback_cases=max(heuristic_fallback_cases, 0),
        missing_benchmark_evidence=missing_benchmark_evidence,
        status=status,
        note=_dataset_health_note(
            status=status,
            planned_verify_cases=planned_verify_cases,
            built_verify_cases=built_verify_cases,
            benchmark_backed_cases=max(benchmark_backed_cases, 0),
            heuristic_fallback_cases=max(heuristic_fallback_cases, 0),
            missing_benchmark_evidence=missing_benchmark_evidence,
        ),
    )


def _build_summary_message(
    *,
    status: str,
    total_verify_cases: int,
    benchmark_backed_case_count: int,
    heuristic_fallback_case_count: int,
    datasets_with_missing_benchmark_evidence: Sequence[str],
) -> str:
    affected_dataset_count = len(list(datasets_with_missing_benchmark_evidence))

    if status == "healthy":
        return f"All {total_verify_cases} curated verify case(s) are backed by benchmark evidence."

    if status == "incomplete":
        return (
            f"Built {total_verify_cases} curated verify case(s); "
            "the demo catalog is still incomplete."
        )

    if heuristic_fallback_case_count > 0:
        if affected_dataset_count > 0:
            return (
                f"{heuristic_fallback_case_count} of {total_verify_cases} curated verify case(s) "
                f"use heuristic fallback across {affected_dataset_count} dataset(s)."
            )
        return (
            f"{heuristic_fallback_case_count} of {total_verify_cases} curated verify case(s) "
            "use heuristic fallback."
        )

    return f"Benchmark evidence is incomplete for {affected_dataset_count} dataset(s)."


def build_catalog_build_health_summary(
    raw_catalog_build_health: Any,
    *,
    case_evidence: Sequence[EvidenceQualitySummary] = (),
    dataset_demo_health: Mapping[str, CatalogDatasetDemoHealth] | None = None,
) -> CatalogBuildHealthSummary | None:
    raw_health = _mapping_from(raw_catalog_build_health)
    dataset_demo_health = dataset_demo_health or {}

    if raw_health:
        driver_counts = _mapping_from(raw_health.get("case_selection_driver_counts"))
        benchmark_backed_case_count = _normalize_count(driver_counts.get("benchmark_driven"))
        heuristic_fallback_case_count = _normalize_count(driver_counts.get("heuristic_fallback")) + _normalize_count(
            driver_counts.get("heuristic_only")
        )
        total_verify_cases = (
            _normalize_count(raw_health.get("total_verify_cases_built"))
            or _normalize_count(raw_health.get("total_verify_cases_planned"))
        )
        if total_verify_cases == 0:
            total_verify_cases = benchmark_backed_case_count + heuristic_fallback_case_count

        datasets_with_missing_benchmark_evidence = sorted(
            {
                dataset_key
                for dataset_key in (_dataset_key_only(value) for value in raw_health.get("datasets_with_missing_benchmark_evidence") or [])
                if dataset_key
            }
        )
        status = _normalize_text(raw_health.get("status")) or "healthy"
        return CatalogBuildHealthSummary(
            catalog_build_status=status,
            total_verify_cases=total_verify_cases,
            benchmark_backed_case_count=benchmark_backed_case_count,
            heuristic_fallback_case_count=heuristic_fallback_case_count,
            datasets_with_missing_benchmark_evidence=datasets_with_missing_benchmark_evidence,
            summary_message=_build_summary_message(
                status=status,
                total_verify_cases=total_verify_cases,
                benchmark_backed_case_count=benchmark_backed_case_count,
                heuristic_fallback_case_count=heuristic_fallback_case_count,
                datasets_with_missing_benchmark_evidence=datasets_with_missing_benchmark_evidence,
            ),
        )

    evidence_items = list(case_evidence)
    if not evidence_items and not dataset_demo_health:
        return None

    total_verify_cases = len(evidence_items)
    benchmark_backed_case_count = sum(1 for item in evidence_items if item.benchmark_backed_selection)
    heuristic_fallback_case_count = sum(1 for item in evidence_items if item.selection_driver == "heuristic_fallback")
    datasets_with_missing_benchmark_evidence = sorted(
        dataset
        for dataset, health in dataset_demo_health.items()
        if health.missing_benchmark_evidence
    )

    if any(health.status == "incomplete" for health in dataset_demo_health.values()):
        status = "incomplete"
    elif heuristic_fallback_case_count > 0 or datasets_with_missing_benchmark_evidence:
        status = "degraded"
    else:
        status = "healthy"

    return CatalogBuildHealthSummary(
        catalog_build_status=status,
        total_verify_cases=total_verify_cases,
        benchmark_backed_case_count=benchmark_backed_case_count,
        heuristic_fallback_case_count=heuristic_fallback_case_count,
        datasets_with_missing_benchmark_evidence=datasets_with_missing_benchmark_evidence,
        summary_message=_build_summary_message(
            status=status,
            total_verify_cases=total_verify_cases,
            benchmark_backed_case_count=benchmark_backed_case_count,
            heuristic_fallback_case_count=heuristic_fallback_case_count,
            datasets_with_missing_benchmark_evidence=datasets_with_missing_benchmark_evidence,
        ),
    )
