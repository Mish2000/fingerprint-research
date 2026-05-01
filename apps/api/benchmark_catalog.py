from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from apps.api.benchmark_meta import (
    CANONICAL_FULL_RUNS,
    DATASET_ORDER,
    SHOWCASE_DATASETS,
    SPLIT_ORDER,
    benchmark_row_is_current,
    benchmark_method_sort_key,
    benchmark_method_to_canonical,
    canonical_method_label,
    canonical_method_label_from_benchmark,
    canonical_method_sort_key,
    dataset_meta,
    default_split_for_view,
    excluded_run_name,
    format_run_label,
    infer_dataset_from_run,
    infer_run_kind,
    infer_view_mode,
    is_showcase_dataset,
    path_is_under,
    run_sort_key,
    split_meta,
    view_mode_meta,
)
from apps.api.schemas import (
    BenchmarkArtifactLink,
    BenchmarkProvenance,
    BenchmarkRunInfo,
    BenchmarkRunsResponse,
    BenchmarkSummaryResponse,
    BestMethodEntry,
    BestMethodsResponse,
    ComparisonResponse,
    ComparisonRow,
    NamedInfo,
)

ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = ROOT / "artifacts" / "reports" / "benchmark"

SORT_MODES = {"best_accuracy", "lowest_eer", "lowest_latency"}
VIEW_MODES = ("canonical", "smoke", "archive")
SHOWCASE_VIEW_MODE = "canonical"
SHOWCASE_METRICS = ("best_auc", "best_eer", "best_latency")
SHOWCASE_SECONDARY_EVIDENCE_KEYS = {"scores_csv", "meta_json", "roc_png", "markdown_summary", "run_manifest"}

SELECTION_POLICY_NOTE = (
    "Curated full benchmark showcase restricted to validated canonical families with comparison-ready rows, "
    "deterministic champion metrics, and usable evidence."
)

ARTIFACT_NOTE = (
    "Artifact links surface stored benchmark evidence when files are available. Missing files remain non-fatal "
    "so the showcase stays stable."
)

_ARTIFACT_LABELS = {
    "summary_csv": "Summary CSV",
    "scores_csv": "Scores CSV",
    "meta_json": "Meta JSON",
    "roc_png": "ROC Preview",
    "markdown_summary": "Markdown Summary",
    "run_manifest": "Run Manifest",
    "run_log": "Run Log",
}


def _named_info(key: str, factory) -> Optional[NamedInfo]:
    meta = factory(key)
    if meta is None:
        return None
    return NamedInfo(key=key, label=meta["label"], summary=meta.get("summary", ""))


def _dataset_info(key: str) -> Optional[NamedInfo]:
    return _named_info(key, dataset_meta)


def _split_info(key: str) -> Optional[NamedInfo]:
    return _named_info(key, split_meta)


def _view_info(key: str) -> Optional[NamedInfo]:
    return _named_info(key, view_mode_meta)


def _maybe_float(raw: object) -> Optional[float]:
    if raw in ("", None):
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return float(text)


def _maybe_int(raw: object) -> Optional[int]:
    if raw in ("", None):
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return int(float(text))


def _parse_json_text(raw: object) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_read_json(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _relative_filename(run_dir: Path, path: Path | None) -> Optional[str]:
    if path is None or not path.exists():
        return None
    if not path_is_under(run_dir, path):
        return None
    return str(path.resolve().relative_to(run_dir.resolve())).replace("\\", "/")


def _artifact_url(run: str, filename: str | None) -> Optional[str]:
    if not filename:
        return None
    return f"/api/benchmark/artifacts/{run}/{filename}"


def _artifact_link(run: str, run_dir: Path, key: str, path: Path | None) -> BenchmarkArtifactLink:
    filename = _relative_filename(run_dir, path)
    return BenchmarkArtifactLink(
        key=key,
        label=_ARTIFACT_LABELS[key],
        available=filename is not None,
        url=_artifact_url(run, filename),
    )


def _resolve_candidate_path(run_dir: Path, raw_path: object, fallback: Path | None = None) -> Path | None:
    text = str(raw_path or "").strip()
    if text:
        candidate = Path(text)
        if candidate.exists() and path_is_under(run_dir, candidate):
            return candidate
    if fallback is not None and fallback.exists():
        return fallback
    return None


def _run_summary_note(view_mode: str, validated: bool, limit: Optional[int]) -> str:
    if view_mode == "canonical":
        return "Validated canonical benchmark family selected for the main showcase." if validated else (
            "Canonical candidate is missing validation and was downgraded from the primary showcase."
        )
    if view_mode == "smoke":
        if limit and limit > 0:
            return f"Smoke regression anchor limited to {limit} pairs per method and kept separate from full benchmarks."
        return "Smoke regression anchor kept separate from the main full-benchmark story."
    return "Archived or non-primary run kept for provenance and historical comparison only."


def _row_summary_text(
    *,
    view_mode: str,
    validation_state: str,
    run_label: str,
    n_pairs: Optional[int],
    latency_source: Optional[str],
) -> str:
    prefix = {
        "validated": "Validated showcase row",
        "snapshot": "Smoke snapshot row",
        "archived": "Archived comparison row",
        "partial": "Partial comparison row",
    }.get(validation_state, "Benchmark row")
    pair_text = f"{n_pairs:,} pairs" if isinstance(n_pairs, int) and n_pairs > 0 else "pair count unavailable"
    latency_text = {
        "reported": "reported latency",
        "wall": "wall-clock latency",
        None: "latency N/A",
    }[latency_source]
    return f"{prefix} from {run_label}. {pair_text}. Uses {latency_text}."


def _validation_state_for_run(view_mode: str, validated: bool, partial: bool) -> str:
    if partial:
        return "partial"
    if view_mode == "canonical":
        return "validated" if validated else "partial"
    if view_mode == "smoke":
        return "snapshot"
    return "archived"


def _status_for_run(view_mode: str, validated: bool, partial: bool) -> str:
    if partial:
        return "partial"
    if view_mode == "canonical":
        return "validated" if validated else "partial"
    if view_mode == "smoke":
        return "smoke"
    return "archived"


def _tie_break_key(row: ComparisonRow) -> Tuple[Any, ...]:
    return (
        benchmark_method_sort_key(row.benchmark_method),
        run_sort_key(row.run, validated=row.validation_state == "validated"),
        row.method,
        row.benchmark_method,
        row.run,
        row.split,
    )


def _rank_rows(rows: List[ComparisonRow]) -> None:
    auc_rows = sorted(
        rows,
        key=lambda row: (
            -row.auc,
            row.eer,
            float("inf") if row.latency_ms is None else row.latency_ms,
            _tie_break_key(row),
        ),
    )
    eer_rows = sorted(
        rows,
        key=lambda row: (
            row.eer,
            -row.auc,
            float("inf") if row.latency_ms is None else row.latency_ms,
            _tie_break_key(row),
        ),
    )
    latency_rows = sorted(
        rows,
        key=lambda row: (
            float("inf") if row.latency_ms is None else row.latency_ms,
            -row.auc,
            row.eer,
            _tie_break_key(row),
        ),
    )

    for index, row in enumerate(auc_rows, start=1):
        row.auc_rank = index
    for index, row in enumerate(eer_rows, start=1):
        row.eer_rank = index
    for index, row in enumerate(latency_rows, start=1):
        row.latency_rank = None if row.latency_ms is None else index


def _sort_rows(rows: Iterable[ComparisonRow], sort_mode: str) -> List[ComparisonRow]:
    if sort_mode == "lowest_eer":
        return sorted(
            rows,
            key=lambda row: (
                row.eer_rank if row.eer_rank is not None else 999,
                row.auc_rank if row.auc_rank is not None else 999,
                _tie_break_key(row),
            ),
        )
    if sort_mode == "lowest_latency":
        return sorted(
            rows,
            key=lambda row: (
                row.latency_rank if row.latency_rank is not None else 999,
                row.auc_rank if row.auc_rank is not None else 999,
                _tie_break_key(row),
            ),
        )
    return sorted(
        rows,
        key=lambda row: (
            row.auc_rank if row.auc_rank is not None else 999,
            row.eer_rank if row.eer_rank is not None else 999,
            _tie_break_key(row),
        ),
    )


def _available_artifact_keys(links: Iterable[BenchmarkArtifactLink]) -> List[str]:
    return [item.key for item in links if item.available]


def _primary_meta_path(run_dir: Path, method: str, split: str, raw_meta: object) -> Path | None:
    candidates = [
        run_dir / f"run_{method}_{split}.meta.json",
        _resolve_candidate_path(run_dir, raw_meta),
        run_dir / f"scores_{method}_{split}.meta.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _build_available_dataset_infos() -> List[NamedInfo]:
    return [_dataset_info(dataset) for dataset in SHOWCASE_DATASETS if _dataset_info(dataset) is not None]


def _build_dataset_infos(dataset_keys: Iterable[str]) -> List[NamedInfo]:
    return [info for dataset in dataset_keys if (info := _dataset_info(dataset)) is not None]


def _build_split_infos(split_keys: Iterable[str]) -> List[NamedInfo]:
    return [info for split in split_keys if (info := _split_info(split)) is not None]


def _read_summary_csv(summary_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(dict(raw))
    return rows


def _dataset_name(raw: object) -> Optional[str]:
    if isinstance(raw, dict):
        name = str(raw.get("name", "")).strip().lower()
        return name or None
    text = str(raw or "").strip().lower()
    return text or None


def _infer_dataset(run: str, manifest: Dict[str, Any], summary_rows: List[Dict[str, Any]]) -> Optional[str]:
    dataset = infer_dataset_from_run(run)
    if dataset:
        return dataset
    manifest_dataset = _dataset_name(manifest.get("dataset"))
    if manifest_dataset:
        return manifest_dataset
    for row in summary_rows:
        config = _parse_json_text(row.get("config_json"))
        config_dataset = _dataset_name(config.get("dataset"))
        if config_dataset:
            return config_dataset
    return None


def _scan_runs(root: Path = BENCH_ROOT) -> List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]]:
    if not root.exists():
        return []

    catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]] = []

    for run_dir in sorted(root.iterdir(), key=lambda item: item.name):
        if not run_dir.is_dir():
            continue
        if excluded_run_name(run_dir.name):
            continue

        summary_csv = run_dir / "results_summary.csv"
        if not summary_csv.exists():
            continue

        try:
            raw_rows = _read_summary_csv(summary_csv)
        except OSError:
            continue

        manifest_payload = _safe_read_json(run_dir / "run_manifest.json")
        dataset = _infer_dataset(run_dir.name, manifest_payload, raw_rows)
        if not is_showcase_dataset(dataset):
            continue

        validated = (run_dir / "validation.ok").exists()
        run_kind = infer_run_kind(run_dir.name)
        view_mode = infer_view_mode(run_dir.name, validated=validated)
        limit = _maybe_int(manifest_payload.get("limit"))
        current_rows = [
            raw_row
            for raw_row in raw_rows
            if benchmark_row_is_current(str(raw_row.get("method", "")).strip(), raw_row)
        ]
        methods_in_run = sorted(
            {
                str(row.get("method", "")).strip()
                for row in current_rows
                if str(row.get("method", "")).strip()
            },
            key=benchmark_method_sort_key,
        )
        canonical_methods_in_run = sorted(
            {
                benchmark_method_to_canonical(method)
                for method in methods_in_run
                if benchmark_method_to_canonical(method)
            },
            key=canonical_method_sort_key,
        )

        comparison_rows: List[ComparisonRow] = []
        methods: set[str] = set()
        benchmark_methods: set[str] = set()
        splits: set[str] = set()

        for raw_row in current_rows:
            benchmark_method = str(raw_row.get("method", "")).strip()
            split = str(raw_row.get("split", "")).strip()
            auc = _maybe_float(raw_row.get("auc"))
            eer = _maybe_float(raw_row.get("eer"))
            if not benchmark_method or not split or auc is None or eer is None:
                continue

            method = benchmark_method_to_canonical(benchmark_method)
            method_label = canonical_method_label_from_benchmark(benchmark_method)

            methods.add(method)
            benchmark_methods.add(benchmark_method)
            splits.add(split)

            scores_csv = _resolve_candidate_path(
                run_dir,
                raw_row.get("scores_csv"),
                run_dir / f"scores_{benchmark_method}_{split}.csv",
            )
            meta_json = _primary_meta_path(run_dir, benchmark_method, split, raw_row.get("meta_json"))
            method_meta = _safe_read_json(meta_json)
            config_json = _parse_json_text(raw_row.get("config_json"))

            latency_reported = _maybe_float(raw_row.get("avg_ms_pair_reported"))
            latency_wall = _maybe_float(raw_row.get("avg_ms_pair_wall"))
            latency_ms = latency_reported if latency_reported is not None else latency_wall
            latency_source = "reported" if latency_reported is not None else ("wall" if latency_wall is not None else None)

            artifacts = [
                _artifact_link(run_dir.name, run_dir, "summary_csv", summary_csv),
                _artifact_link(run_dir.name, run_dir, "scores_csv", scores_csv),
                _artifact_link(run_dir.name, run_dir, "meta_json", meta_json),
                _artifact_link(run_dir.name, run_dir, "roc_png", run_dir / f"roc_{benchmark_method}_{split}.png"),
                _artifact_link(run_dir.name, run_dir, "markdown_summary", run_dir / "results_summary.md"),
                _artifact_link(run_dir.name, run_dir, "run_manifest", run_dir / "run_manifest.json"),
                _artifact_link(run_dir.name, run_dir, "run_log", run_dir / "run.log"),
            ]
            available_artifacts = _available_artifact_keys(artifacts)

            partial = len(available_artifacts) < 3
            validation_state = _validation_state_for_run(view_mode, validated=validated, partial=partial)
            status = _status_for_run(view_mode, validated=validated, partial=partial)
            run_label = format_run_label(run_dir.name)

            provenance = BenchmarkProvenance(
                run=run_dir.name,
                run_label=run_label,
                run_kind=run_kind,
                view_mode=view_mode,
                status=status,
                validation_state=validation_state,
                artifact_source="results_summary.csv",
                methods_in_run=canonical_methods_in_run,
                benchmark_methods_in_run=methods_in_run,
                canonical_method=method,
                benchmark_method=benchmark_method,
                method_label=method_label,
                timestamp_utc=str(raw_row.get("timestamp_utc", "")).strip() or manifest_payload.get("timestamp_utc"),
                limit=limit,
                pairs_path=str(method_meta.get("pairs_path") or config_json.get("pairs_path") or "").strip() or None,
                manifest_path=str(method_meta.get("manifest_path") or config_json.get("manifest_path") or "").strip() or None,
                data_dir=str(method_meta.get("resolved_data_dir") or config_json.get("resolved_data_dir") or manifest_payload.get("data_dir") or "").strip() or None,
                git_commit=str(manifest_payload.get("git", {}).get("commit") or "").strip() or None,
                available_artifacts=available_artifacts,
            )

            comparison_rows.append(
                ComparisonRow(
                    dataset=dataset,
                    run=run_dir.name,
                    split=split,
                    method=method,
                    benchmark_method=benchmark_method,
                    method_label=method_label,
                    auc=auc,
                    eer=eer,
                    n_pairs=_maybe_int(raw_row.get("n_pairs")),
                    tar_at_far_1e_2=_maybe_float(raw_row.get("tar_at_far_1e_2")),
                    tar_at_far_1e_3=_maybe_float(raw_row.get("tar_at_far_1e_3")),
                    latency_ms=latency_ms,
                    latency_source=latency_source,
                    run_family=run_dir.name,
                    run_label=run_label,
                    run_kind=run_kind,
                    view_mode=view_mode,
                    status=status,
                    validation_state=validation_state,
                    artifact_count=len(available_artifacts),
                    available_artifacts=available_artifacts,
                    summary_text=_row_summary_text(
                        view_mode=view_mode,
                        validation_state=validation_state,
                        run_label=run_label,
                        n_pairs=_maybe_int(raw_row.get("n_pairs")),
                        latency_source=latency_source,
                    ),
                    artifacts=artifacts,
                    provenance=provenance,
                )
            )

        if not comparison_rows:
            continue

        methods_sorted = sorted(methods, key=canonical_method_sort_key)
        benchmark_methods_sorted = sorted(benchmark_methods, key=benchmark_method_sort_key)
        splits_sorted = sorted(splits, key=lambda item: SPLIT_ORDER.get(item, 99))
        partial_run = any(row.status == "partial" for row in comparison_rows)
        run_info = BenchmarkRunInfo(
            run=run_dir.name,
            dataset=dataset,
            run_kind=run_kind,
            view_mode=view_mode,
            status=_status_for_run(view_mode, validated=validated, partial=partial_run),
            validation_state=_validation_state_for_run(view_mode, validated=validated, partial=partial_run),
            validated=validated,
            recommended=run_dir.name in CANONICAL_FULL_RUNS and validated,
            run_label=format_run_label(run_dir.name),
            artifact_count=max((row.artifact_count for row in comparison_rows), default=0),
            summary_note=_run_summary_note(view_mode, validated=validated, limit=limit),
            methods=methods_sorted,
            benchmark_methods=benchmark_methods_sorted,
            splits=splits_sorted,
            dataset_info=_dataset_info(dataset),
        )
        catalog.append((run_info, comparison_rows))

    catalog.sort(key=lambda item: run_sort_key(item[0].run, validated=item[0].validated))
    return catalog


def _selection_rows(
    catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]],
    *,
    dataset: str,
    split: str,
    view_mode: str,
) -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []
    for run_info, run_rows in catalog:
        if run_info.view_mode != view_mode:
            continue
        if run_info.dataset != dataset:
            continue
        for row in run_rows:
            if split != "all" and row.split != split:
                continue
            rows.append(row.model_copy(deep=True))
    _rank_rows(rows)
    return rows


def _row_has_showcase_evidence(row: ComparisonRow) -> bool:
    available = set(row.available_artifacts)
    return "summary_csv" in available and bool(available & SHOWCASE_SECONDARY_EVIDENCE_KEYS)


def _best_entries_from_rows(rows: List[ComparisonRow]) -> List[BestMethodEntry]:
    best_entries: List[BestMethodEntry] = []
    metrics = (
        ("best_auc", lambda row: row.auc_rank if row.auc_rank is not None else 999),
        ("best_eer", lambda row: row.eer_rank if row.eer_rank is not None else 999),
        ("best_latency", lambda row: row.latency_rank if row.latency_rank is not None else 999),
    )

    for metric, rank_accessor in metrics:
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                rank_accessor(row),
                _tie_break_key(row),
            ),
        )
        if metric == "best_latency":
            sorted_rows = [row for row in sorted_rows if row.latency_ms is not None]
        if not sorted_rows:
            continue

        winner = sorted_rows[0]
        value = winner.auc
        if metric == "best_eer":
            value = winner.eer
        elif metric == "best_latency":
            value = float(winner.latency_ms or 0.0)

        best_entries.append(
            BestMethodEntry(
                dataset=winner.dataset,
                split=winner.split,
                metric=metric,
                method=winner.method,
                benchmark_method=winner.benchmark_method,
                method_label=winner.method_label or canonical_method_label(winner.method),
                run=winner.run,
                value=value,
                run_family=winner.run_family,
                run_label=winner.run_label,
                view_mode=winner.view_mode,
                status=winner.status,
                validation_state=winner.validation_state,
            )
        )

    best_entries.sort(
        key=lambda entry: (
            DATASET_ORDER.get(entry.dataset, 99),
            SPLIT_ORDER.get(entry.split, 99),
            {"best_auc": 0, "best_eer": 1, "best_latency": 2}.get(entry.metric, 99),
            benchmark_method_sort_key(entry.benchmark_method or entry.method),
        )
    )
    return best_entries


def _selection_is_showcase_ready(rows: List[ComparisonRow]) -> bool:
    if not rows:
        return False
    if not any(_row_has_showcase_evidence(row) for row in rows):
        return False
    best_metrics = {entry.metric for entry in _best_entries_from_rows(rows)}
    return all(metric in best_metrics for metric in SHOWCASE_METRICS)


def _available_splits_for_selection(
    catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]],
    *,
    dataset: str,
    view_mode: str,
) -> List[str]:
    splits = {
        row.split
        for run_info, run_rows in catalog
        if run_info.view_mode == view_mode and run_info.dataset == dataset
        for row in run_rows
    }
    return sorted(splits, key=lambda item: SPLIT_ORDER.get(item, 99))


def _showcase_splits_for_dataset(
    catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]],
    dataset: str,
) -> List[str]:
    valid_splits: List[str] = []
    for split in _available_splits_for_selection(catalog, dataset=dataset, view_mode=SHOWCASE_VIEW_MODE):
        rows = _selection_rows(catalog, dataset=dataset, split=split, view_mode=SHOWCASE_VIEW_MODE)
        if _selection_is_showcase_ready(rows):
            valid_splits.append(split)
    return valid_splits


def _showcase_dataset_keys(catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]]) -> List[str]:
    return [
        dataset
        for dataset in SHOWCASE_DATASETS
        if _showcase_splits_for_dataset(catalog, dataset)
    ]


def _resolve_dataset(
    catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]],
    dataset: Optional[str],
) -> str:
    available = _showcase_dataset_keys(catalog)
    if dataset in available:
        return str(dataset)
    if available:
        return available[0]
    return SHOWCASE_DATASETS[0]


def _resolve_split(
    catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]],
    dataset: str,
    split: Optional[str],
) -> str:
    available = _showcase_splits_for_dataset(catalog, dataset)
    if split and split in available:
        return split
    default_split = default_split_for_view(available, SHOWCASE_VIEW_MODE)
    return default_split or "test"


def _selection_validation_state(rows: List[ComparisonRow], view_mode: str) -> str:
    if not rows:
        return {
            "canonical": "validated",
            "smoke": "snapshot",
            "archive": "archived",
        }.get(view_mode, "snapshot")
    states = {row.validation_state for row in rows}
    if "partial" in states:
        return "partial"
    if view_mode == "canonical":
        return "validated"
    if view_mode == "smoke":
        return "snapshot"
    return "archived"


def _selection_note(view_mode: str, rows: List[ComparisonRow]) -> str:
    base = {
        "canonical": "Showing curated full benchmark results from validated showcase runs.",
        "smoke": "Showing curated full benchmark results from validated showcase runs.",
        "archive": "Showing curated full benchmark results from validated showcase runs.",
    }.get(view_mode, "Showing curated full benchmark results from validated showcase runs.")
    if rows:
        return base
    return "No curated full benchmark results are available for the current selection."


def load_benchmark_runs(root: Path = BENCH_ROOT) -> BenchmarkRunsResponse:
    catalog = _scan_runs(root)
    items = [item for item, _ in catalog]
    showcase_datasets = _showcase_dataset_keys(catalog)

    default_run = next((item.run for item in items if item.recommended), None)
    if default_run is None and items:
        default_run = items[0].run
    default_dataset = None
    default_split = None

    if showcase_datasets:
        default_dataset = showcase_datasets[0]
    elif items:
        default_dataset = next((item.dataset for item in items if item.view_mode == SHOWCASE_VIEW_MODE and item.dataset), None)
        if default_dataset is None:
            default_dataset = items[0].dataset
    if default_dataset:
        default_split = _resolve_split(catalog, default_dataset, None)

    return BenchmarkRunsResponse(
        default_run=default_run,
        default_dataset=default_dataset,
        default_split=default_split,
        default_view_mode="canonical",
        runs=items,
    )


def load_benchmark_summary(
    *,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    view_mode: str = "canonical",
    root: Path = BENCH_ROOT,
) -> BenchmarkSummaryResponse:
    normalized_view_mode = SHOWCASE_VIEW_MODE
    catalog = _scan_runs(root)
    available_dataset_keys = _showcase_dataset_keys(catalog)
    resolved_dataset = _resolve_dataset(catalog, dataset)
    available_split_keys = _showcase_splits_for_dataset(catalog, resolved_dataset)
    resolved_split = _resolve_split(catalog, resolved_dataset, split)
    rows = _selection_rows(catalog, dataset=resolved_dataset, split=resolved_split, view_mode=normalized_view_mode)
    current_runs = sorted({row.run for row in rows}, key=lambda run: run_sort_key(run, validated=True))

    return BenchmarkSummaryResponse(
        dataset=resolved_dataset,
        split=resolved_split,
        view_mode=normalized_view_mode,
        dataset_info=_dataset_info(resolved_dataset),
        split_info=_split_info(resolved_split),
        view_info=_view_info(normalized_view_mode),
        validation_state=_selection_validation_state(rows, normalized_view_mode),
        selection_note=_selection_note(normalized_view_mode, rows),
        selection_policy=SELECTION_POLICY_NOTE,
        result_count=len(rows),
        method_count=len({row.method for row in rows}),
        run_count=len(current_runs),
        available_datasets=_build_dataset_infos(available_dataset_keys),
        available_splits=_build_split_infos(available_split_keys),
        available_view_modes=[info for key in [SHOWCASE_VIEW_MODE] if (info := _view_info(key)) is not None],
        current_run_families=current_runs,
        artifact_note=ARTIFACT_NOTE,
    )


def load_comparison(
    *,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    view_mode: str = "canonical",
    sort_mode: str = "best_accuracy",
    root: Path = BENCH_ROOT,
) -> ComparisonResponse:
    normalized_view_mode = SHOWCASE_VIEW_MODE
    normalized_sort_mode = sort_mode if sort_mode in SORT_MODES else "best_accuracy"
    catalog = _scan_runs(root)
    resolved_dataset = _resolve_dataset(catalog, dataset)
    resolved_split = _resolve_split(catalog, resolved_dataset, split)
    rows = _selection_rows(catalog, dataset=resolved_dataset, split=resolved_split, view_mode=normalized_view_mode)
    rows = _sort_rows(rows, normalized_sort_mode)

    dataset_info = {
        dataset_key: info
        for dataset_key in _showcase_dataset_keys(catalog)
        if (info := _dataset_info(dataset_key)) is not None
    }
    split_keys = _showcase_splits_for_dataset(catalog, resolved_dataset)

    return ComparisonResponse(
        rows=rows,
        datasets=list(dataset_info.keys()),
        splits=split_keys,
        default_dataset=resolved_dataset,
        default_split=resolved_split,
        view_mode=normalized_view_mode,
        view_info=_view_info(normalized_view_mode),
        dataset_info=dataset_info,
        split_info={key: info for key in ["all", *split_keys] if (info := _split_info(key)) is not None},
    )


def load_best_methods(
    *,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    view_mode: str = "canonical",
    root: Path = BENCH_ROOT,
) -> BestMethodsResponse:
    normalized_view_mode = SHOWCASE_VIEW_MODE
    catalog = _scan_runs(root)
    resolved_dataset = _resolve_dataset(catalog, dataset)
    resolved_split = _resolve_split(catalog, resolved_dataset, split)
    rows = _selection_rows(catalog, dataset=resolved_dataset, split=resolved_split, view_mode=normalized_view_mode)
    best_entries = _best_entries_from_rows(rows)

    return BestMethodsResponse(
        dataset=resolved_dataset,
        split=resolved_split,
        view_mode=normalized_view_mode,
        entries=best_entries,
    )


def resolve_benchmark_artifact(run: str, filename: str, root: Path = BENCH_ROOT) -> Path:
    if not run or not filename:
        raise FileNotFoundError("Benchmark artifact request is missing run or filename.")
    run_dir = root / run
    if not run_dir.exists():
        raise FileNotFoundError(f"Unknown benchmark run: {run}")
    target = (run_dir / filename).resolve()
    if not path_is_under(run_dir, target):
        raise FileNotFoundError("Benchmark artifact path escaped the run directory.")
    if not target.exists():
        raise FileNotFoundError(f"Missing benchmark artifact: {target}")
    return target
