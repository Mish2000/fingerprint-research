from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from apps.api.benchmark_meta import (
    ACCEPTED_CANONICAL_FULL_RUNS,
    ACCEPTED_CANONICAL_SMOKE_RUNS,
    CANONICAL_FULL_RUNS,
    CANONICAL_SMOKE_RUNS,
    REFERENCE_CANONICAL_FULL_RUNS,
    REFERENCE_CANONICAL_SMOKE_RUNS,
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
    method_presentation_metadata,
    method_research_track,
    method_showcase_eligible,
    method_showcase_exclusion_note,
    path_is_under,
    run_sort_key,
    split_meta,
    view_mode_meta,
)
from apps.api.schemas import (
    BenchmarkArtifactLink,
    BenchmarkOperatingPoint,
    BenchmarkProvenance,
    BenchmarkRunInfo,
    BenchmarkRunsResponse,
    BenchmarkSummaryResponse,
    BenchmarkTarFarDistributionRow,
    BestMethodEntry,
    BestMethodsResponse,
    ComparisonResponse,
    ComparisonRow,
    NamedInfo,
)

ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = ROOT / "artifacts" / "reports" / "benchmark"
BENCH_REFERENCE_ROOT = ROOT / "archive" / "reports" / "benchmark_reference"

SORT_MODES = {"best_accuracy", "lowest_eer", "lowest_latency"}
VIEW_MODES = ("canonical", "smoke", "archive")
SHOWCASE_VIEW_MODE = "canonical"
SHOWCASE_METRICS = ("best_auc", "best_eer", "best_latency")
SHOWCASE_SECONDARY_EVIDENCE_KEYS = {"scores_csv", "meta_json", "roc_png", "markdown_summary", "final_markdown", "run_manifest"}
CURATED_REFERENCE_RUNS = set(ACCEPTED_CANONICAL_FULL_RUNS) | set(ACCEPTED_CANONICAL_SMOKE_RUNS)
PLAIN_ROLL_FINAL_BUNDLE = "plain_roll_final_sift_v1"
SOURCEAFIS_FINAL_BUNDLE = "plain_roll_final_sourceafis_v1"
PLAIN_ROLL_FINAL_DATASETS = ("nist_sd300b", "nist_sd300c")
PLAIN_ROLL_FINAL_METHODS = ("sift_plain_roll_v2", "sift", "minutiae", "harris", "classic_v2")
SOURCEAFIS_FINAL_METHODS = ("sourceafis_open",)
PLAIN_ROLL_FINAL_DISPLAY_ORDER = {
    "sourceafis_open": 0,
    "sift_plain_roll_v2": 1,
    "sift": 2,
    "minutiae": 3,
    "harris": 4,
    "classic_v2": 5,
}
PLAIN_ROLL_FINAL_RUNS = {
    f"{PLAIN_ROLL_FINAL_BUNDLE}_{dataset}_final"
    for dataset in PLAIN_ROLL_FINAL_DATASETS
}
SOURCEAFIS_FINAL_RUNS = {
    f"{SOURCEAFIS_FINAL_BUNDLE}_{dataset}_final"
    for dataset in PLAIN_ROLL_FINAL_DATASETS
}
BENCHMARK_SOURCE_LABELS = {
    "live": "Live artifacts",
    "reference": "Reference artifacts",
}
FINAL_EVIDENCE_SOURCE_LABEL = "Final curated evidence"

DEDICATED_SHOWCASE_EXCLUSION_NOTE = (
    "Dedicated Patch AI remains available as an experimental research method, but the current checkpoint/results do "
    "not satisfy canonical benchmark showcase criteria and must not be promoted as a winner or champion."
)

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
    "final_markdown": "Final Markdown Evidence",
    "run_manifest": "Run Manifest",
    "run_log": "Run Log",
    "thresholds_csv": "Thresholds CSV",
    "threshold_sweep_csv": "Threshold Sweep CSV",
    "tar_far_distribution_csv": "TAR/FAR Distribution CSV",
    "latency_summary": "Latency Summary",
    "positive_only_metrics": "Positive-only Metrics",
    "negative_only_metrics": "Negative-only Metrics",
    "failures_csv": "Failures CSV",
}

SOURCEAFIS_METHOD_LABEL = "SourceAFIS Open Matcher"
SOURCEAFIS_SIDE_CAR_NOTE = (
    "SourceAFIS evidence was produced through the fingerprint-engine HTTP sidecar path and is not a default "
    "interactive runtime benchmark method."
)
SIFT_V2_BASELINE_NOTE = (
    "SIFT Plain/Roll v2 is the strongest custom research baseline with exported latency, "
    "but it remains excluded from champion ranking."
)

PLAIN_ROLL_FINAL_METHOD_ROLES = {
    "sift_plain_roll_v2": SIFT_V2_BASELINE_NOTE,
    "sift": (
        "Standard feature baseline under the strict pair-audited plain-vs-roll protocol; "
        "tracked as baseline evidence and not a champion candidate."
    ),
    "harris": (
        "Weak classical baseline under the strict plain-vs-roll protocol; tracked as baseline evidence "
        "and not a champion candidate."
    ),
    "classic_v2": (
        "Weak classical baseline under the strict plain-vs-roll protocol; tracked as baseline evidence "
        "and not a champion candidate."
    ),
    "minutiae": (
        "Weak custom minutiae baseline under the strict plain-vs-roll protocol; tracked as baseline evidence "
        "and not a champion candidate."
    ),
}

CURATED_FINAL_EVIDENCE: List[Dict[str, Any]] = [
    {
        "run": "sourceafis_sd300b_plain_roll_dpi1000_final",
        "filename": "sourceafis_sd300b_plain_roll_dpi1000_final.md",
        "dataset": "nist_sd300b",
        "split": "test",
        "method": "sourceafis_open",
        "method_label": SOURCEAFIS_METHOD_LABEL,
        "run_label": "SourceAFIS SD300B final evidence (DPI 1000)",
        "summary_prefix": "Final SourceAFIS SD300B plain-vs-roll evidence at explicit 1000 DPI",
        "dpi": 1000,
        "n_pairs": 1400,
        "auc": 0.8902,
        "eer": 0.1700,
        "latency_ms": 272.902,
        "timestamp_utc": "2026-05-25T20:16:41Z",
        "manifest_path": "sourceafis_plain_roll_manifest.json",
        "data_dir": "artifacts/reports/benchmark/sourceafis_open_plain_roll_balanced1400_dpi1000",
        "git_commit": "6d4efecdd84da1bdb1cb385fe284039e1423a2ad",
        "note": SOURCEAFIS_SIDE_CAR_NOTE,
        "recommended": True,
        "sourceafis": True,
        "operating_points": [
            {
                "target_far": 0.01,
                "label": "1.00% FAR",
                "threshold": 14.72326764987426,
                "test_tar": 0.7729,
                "test_far": 0.0086,
                "test_frr": 0.2271,
                "ta": 541,
                "fr": 159,
                "fa": 6,
                "tr": 694,
                "calibration_far": 0.0100,
                "calibration_false_accepts": 7,
                "calibration_negatives": 700,
            },
            {
                "target_far": 0.005,
                "label": "0.50% FAR",
                "threshold": 17.393218350729448,
                "test_tar": 0.7600,
                "test_far": 0.0043,
                "test_frr": 0.2400,
                "ta": 532,
                "fr": 168,
                "fa": 3,
                "tr": 697,
                "calibration_far": 0.0043,
                "calibration_false_accepts": 3,
                "calibration_negatives": 700,
            },
        ],
    },
    {
        "run": "sourceafis_sd300c_plain_roll_dpi2000_final",
        "filename": "sourceafis_sd300c_plain_roll_dpi2000_final.md",
        "dataset": "nist_sd300c",
        "split": "test",
        "method": "sourceafis_open",
        "method_label": SOURCEAFIS_METHOD_LABEL,
        "run_label": "SourceAFIS SD300C final evidence (DPI 2000)",
        "summary_prefix": "Final SourceAFIS SD300C plain-vs-roll evidence at explicit 2000 DPI",
        "dpi": 2000,
        "n_pairs": 1400,
        "auc": 0.8815,
        "eer": 0.1743,
        "latency_ms": 249.966,
        "timestamp_utc": "2026-05-25T21:17:06Z",
        "manifest_path": "sourceafis_plain_roll_manifest.json",
        "data_dir": "artifacts/reports/benchmark/sourceafis_open_sd300c_balanced1400_dpi2000",
        "git_commit": "6d4efecdd84da1bdb1cb385fe284039e1423a2ad",
        "note": SOURCEAFIS_SIDE_CAR_NOTE,
        "recommended": True,
        "sourceafis": True,
        "operating_points": [
            {
                "target_far": 0.01,
                "label": "1.00% FAR",
                "threshold": 14.483463789540309,
                "test_tar": 0.7800,
                "test_far": 0.0129,
                "test_frr": 0.2200,
                "ta": 546,
                "fr": 154,
                "fa": 9,
                "tr": 691,
                "calibration_far": 0.0100,
                "calibration_false_accepts": 7,
                "calibration_negatives": 700,
            },
            {
                "target_far": 0.005,
                "label": "0.50% FAR",
                "threshold": 20.06041975470194,
                "test_tar": 0.7529,
                "test_far": 0.0057,
                "test_frr": 0.2471,
                "ta": 527,
                "fr": 173,
                "fa": 4,
                "tr": 696,
                "calibration_far": 0.0043,
                "calibration_false_accepts": 3,
                "calibration_negatives": 700,
            },
        ],
    },
]

CURATED_FINAL_RUNS = {str(item["run"]) for item in CURATED_FINAL_EVIDENCE} | PLAIN_ROLL_FINAL_RUNS | SOURCEAFIS_FINAL_RUNS


def _reference_root_for_live_root(root: Path) -> Path:
    """Return the sibling reference root for the default benchmark layout."""
    try:
        if root.resolve() == BENCH_ROOT.resolve():
            return BENCH_REFERENCE_ROOT
    except OSError:
        pass

    parts = [part.lower() for part in root.parts]
    if len(parts) >= 3 and parts[-3:] == ["artifacts", "reports", "benchmark"]:
        return root.parents[2] / "archive" / "reports" / "benchmark_reference"
    return BENCH_REFERENCE_ROOT


def _benchmark_sources(
    root: Path = BENCH_ROOT,
    reference_root: Path | None = None,
) -> List[Tuple[str, str, Path]]:
    live_root = root
    resolved_reference_root = reference_root if reference_root is not None else _reference_root_for_live_root(root)
    sources = [("live", BENCHMARK_SOURCE_LABELS["live"], live_root)]

    try:
        same_root = live_root.resolve() == resolved_reference_root.resolve()
    except OSError:
        same_root = False
    if not same_root:
        sources.append(("reference", BENCHMARK_SOURCE_LABELS["reference"], resolved_reference_root))
    return sources


def _normalize_view_mode(view_mode: str | None) -> str:
    return view_mode if view_mode in VIEW_MODES else SHOWCASE_VIEW_MODE


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


def _is_available_artifact_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _relative_filename(run_dir: Path, path: Path | None) -> Optional[str]:
    if not _is_available_artifact_file(path):
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


def _loose_artifact_link(run: str, source_root: Path, key: str, filename: str) -> BenchmarkArtifactLink:
    path = (source_root / filename).resolve()
    available = _is_available_artifact_file(path) and path_is_under(source_root, path)
    return BenchmarkArtifactLink(
        key=key,
        label=_ARTIFACT_LABELS[key],
        available=available,
        url=_artifact_url(run, filename) if available else None,
    )


def _source_artifact_link(run: str, source_root: Path, key: str, path: Path | None) -> BenchmarkArtifactLink:
    filename = None
    if _is_available_artifact_file(path):
        resolved_path = path.resolve()
        if path_is_under(source_root, resolved_path):
            filename = str(resolved_path.relative_to(source_root.resolve())).replace("\\", "/")
    return BenchmarkArtifactLink(
        key=key,
        label=_ARTIFACT_LABELS[key],
        available=filename is not None,
        url=_artifact_url(run, filename) if filename is not None else None,
    )


def _resolve_candidate_path(run_dir: Path, raw_path: object, fallback: Path | None = None) -> Path | None:
    text = str(raw_path or "").strip()
    if text:
        candidate = Path(text)
        if _is_available_artifact_file(candidate) and path_is_under(run_dir, candidate):
            return candidate
    if _is_available_artifact_file(fallback):
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


def _showcase_exclusion_note(canonical_method: str, benchmark_method: str) -> Optional[str]:
    return (
        method_showcase_exclusion_note(benchmark_method)
        or method_showcase_exclusion_note(canonical_method)
        or (
            DEDICATED_SHOWCASE_EXCLUSION_NOTE
            if canonical_method == "dedicated" or benchmark_method == "dedicated"
            else None
        )
    )


def _showcase_selection_tier(row: ComparisonRow) -> Tuple[int, str]:
    source = row.provenance.benchmark_source_root if row.provenance else None
    if row.view_mode == "canonical":
        if row.run in CURATED_FINAL_RUNS:
            return (0 if source == "live" else 1, row.run)
        if row.run in CANONICAL_FULL_RUNS:
            return (0 if source == "live" else 1, row.run)
        if row.run in REFERENCE_CANONICAL_FULL_RUNS:
            return (2, row.run)
    if row.view_mode == "smoke":
        if row.run in CANONICAL_SMOKE_RUNS:
            return (0 if source == "live" else 1, row.run)
        if row.run in REFERENCE_CANONICAL_SMOKE_RUNS:
            return (2, row.run)
    return (9, row.run)


def _filter_preferred_showcase_rows(rows: List[ComparisonRow], view_mode: str) -> List[ComparisonRow]:
    if view_mode not in {"canonical", "smoke"} or not rows:
        return rows
    best_tier = min(_showcase_selection_tier(row)[0] for row in rows)
    return [row for row in rows if _showcase_selection_tier(row)[0] == best_tier]


def _tie_break_key(row: ComparisonRow) -> Tuple[Any, ...]:
    final_method_order = None
    if row.run in CURATED_FINAL_RUNS or row.run_family == PLAIN_ROLL_FINAL_BUNDLE:
        final_method_order = PLAIN_ROLL_FINAL_DISPLAY_ORDER.get(row.benchmark_method)
    method_sort_key = (
        (final_method_order, row.benchmark_method)
        if final_method_order is not None
        else benchmark_method_sort_key(row.benchmark_method)
    )
    return (
        method_sort_key,
        run_sort_key(row.run, validated=row.validation_state == "validated"),
        row.method,
        row.benchmark_method,
        row.run,
        row.split,
    )


def _row_is_champion_candidate(row: ComparisonRow) -> bool:
    return bool(row.showcase_eligible) and not bool(row.not_champion_candidate)


def _champion_candidate_rows(rows: Iterable[ComparisonRow]) -> List[ComparisonRow]:
    return [row for row in rows if _row_is_champion_candidate(row)]


def _rank_rows(rows: List[ComparisonRow]) -> None:
    for row in rows:
        row.auc_rank = None
        row.eer_rank = None
        row.latency_rank = None

    candidate_rows = _champion_candidate_rows(rows)
    if not candidate_rows:
        return

    auc_rows = sorted(
        candidate_rows,
        key=lambda row: (
            -row.auc,
            row.eer,
            float("inf") if row.latency_ms is None else row.latency_ms,
            _tie_break_key(row),
        ),
    )
    eer_rows = sorted(
        candidate_rows,
        key=lambda row: (
            row.eer,
            -row.auc,
            float("inf") if row.latency_ms is None else row.latency_ms,
            _tie_break_key(row),
        ),
    )
    latency_rows = sorted(
        candidate_rows,
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
        if _is_available_artifact_file(candidate):
            return candidate
    return None


def _build_available_dataset_infos() -> List[NamedInfo]:
    return [_dataset_info(dataset) for dataset in SHOWCASE_DATASETS if _dataset_info(dataset) is not None]


def _build_dataset_infos(dataset_keys: Iterable[str]) -> List[NamedInfo]:
    return [info for dataset in dataset_keys if (info := _dataset_info(dataset)) is not None]


def _build_split_infos(split_keys: Iterable[str]) -> List[NamedInfo]:
    return [info for split in split_keys if (info := _split_info(split)) is not None]


def _legacy_operating_points(raw_row: Dict[str, Any]) -> List[BenchmarkOperatingPoint]:
    points: List[BenchmarkOperatingPoint] = []
    legacy_specs = (
        ("tar_at_far_1e_2", 0.01, "1.00% FAR"),
        ("tar_at_far_1e_3", 0.001, "0.10% FAR"),
    )
    for field, target_far, label in legacy_specs:
        tar = _maybe_float(raw_row.get(field))
        if tar is None:
            continue
        points.append(
            BenchmarkOperatingPoint(
                target_far=target_far,
                label=label,
                test_tar=tar,
            )
        )
    return points


def _curated_operating_points(spec: Dict[str, Any]) -> List[BenchmarkOperatingPoint]:
    return [
        BenchmarkOperatingPoint(**point)
        for point in spec.get("operating_points", [])
        if isinstance(point, dict)
    ]


def _method_metadata_for_final_evidence(benchmark_method: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    if benchmark_method == "sourceafis_open":
        return {
            "method_status": "optional_external",
            "presentation_tier": "production_candidate",
            "showcase_eligible": True,
            "benchmark_default": False,
            "canonical_default": False,
            "research_track": False,
            "not_champion_candidate": False,
            "showcase_exclusion_note": None,
        }
    return method_presentation_metadata(benchmark_method)


def _method_label_for_final_evidence(benchmark_method: str, spec: Dict[str, Any]) -> str:
    label = str(spec.get("method_label") or "").strip()
    if label:
        return label
    return canonical_method_label_from_benchmark(benchmark_method)


def _read_summary_csv(summary_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(dict(raw))
    return rows


def _safe_read_csv(path: Path) -> List[Dict[str, Any]]:
    if not _is_available_artifact_file(path):
        return []
    try:
        return _read_summary_csv(path)
    except OSError:
        return []


def _plain_roll_final_run(dataset: str) -> str:
    return f"{PLAIN_ROLL_FINAL_BUNDLE}_{dataset}_final"


def _sourceafis_final_run(dataset: str) -> str:
    return f"{SOURCEAFIS_FINAL_BUNDLE}_{dataset}_final"


def _plain_roll_final_markdown_filename(dataset: str, method: str) -> str:
    return f"final_markdown/{dataset}_{method}_plain_roll_final.md"


def _target_far_label(target_far: float) -> str:
    if abs(target_far - 0.01) < 1e-12:
        return "1.00% FAR"
    if abs(target_far - 0.005) < 1e-12:
        return "0.50% FAR"
    return f"{target_far:.3g} FAR"


def _final_bundle_path_from_row(source_root: Path, raw_path: object, fallback: Path | None = None) -> Path | None:
    text = str(raw_path or "").strip()
    if text:
        candidate = Path(text)
        if _is_available_artifact_file(candidate) and path_is_under(source_root, candidate):
            return candidate
    if _is_available_artifact_file(fallback) and path_is_under(source_root, fallback.resolve()):
        return fallback
    return None


def _plain_roll_final_failures_empty(path: Path) -> bool:
    if not _is_available_artifact_file(path):
        return False
    return len(_safe_read_csv(path)) == 0


def _plain_roll_final_pair_audit_ok(manifest: Dict[str, Any], dataset: str, split: str) -> bool:
    audits = manifest.get("pair_audits")
    if not isinstance(audits, list):
        return False
    matching = [
        audit
        for audit in audits
        if isinstance(audit, dict)
        and str(audit.get("dataset") or "") == dataset
        and str(audit.get("split") or "") == split
    ]
    return bool(matching) and all(bool(audit.get("pass")) for audit in matching)


def _plain_roll_final_method_metadata(benchmark_method: str) -> Dict[str, Any]:
    presentation_meta = method_presentation_metadata(benchmark_method)
    note = PLAIN_ROLL_FINAL_METHOD_ROLES.get(benchmark_method)
    if benchmark_method == "sift_plain_roll_v2":
        return {
            **presentation_meta,
            "showcase_exclusion_note": note or presentation_meta.get("showcase_exclusion_note"),
        }
    return {
        **presentation_meta,
        "presentation_tier": "baseline",
        "showcase_eligible": True,
        "research_track": False,
        "not_champion_candidate": True,
        "showcase_exclusion_note": note,
    }


def _build_plain_roll_final_operating_points(
    method_rows: Dict[float, Dict[str, Any]],
    threshold_rows: Dict[float, Dict[str, Any]],
) -> List[BenchmarkOperatingPoint]:
    points: List[BenchmarkOperatingPoint] = []
    for target_far in (0.01, 0.005):
        metric_row = method_rows.get(target_far)
        if metric_row is None:
            continue
        threshold_row = threshold_rows.get(target_far, {})
        points.append(
            BenchmarkOperatingPoint(
                target_far=target_far,
                label=_target_far_label(target_far),
                threshold=_maybe_float(metric_row.get("threshold") or threshold_row.get("threshold")),
                test_tar=_maybe_float(metric_row.get("tar")),
                test_far=_maybe_float(metric_row.get("far")),
                test_frr=_maybe_float(metric_row.get("frr")),
                ta=_maybe_int(metric_row.get("ta")),
                fr=_maybe_int(metric_row.get("fr")),
                fa=_maybe_int(metric_row.get("fa")),
                tr=_maybe_int(metric_row.get("tr")),
                calibration_far=_maybe_float(
                    threshold_row.get("calibration_far")
                    or metric_row.get("threshold_val_far")
                ),
                calibration_false_accepts=_maybe_int(
                    threshold_row.get("calibration_false_accepts")
                    or metric_row.get("threshold_val_false_accepts")
                ),
                calibration_negatives=_maybe_int(threshold_row.get("calibration_negative_count")),
                calibration_positives=_maybe_int(threshold_row.get("calibration_positive_count")),
            )
        )
    return points


def _build_plain_roll_final_tar_far_distribution(
    rows: Iterable[Dict[str, Any]],
) -> List[BenchmarkTarFarDistributionRow]:
    distribution: List[BenchmarkTarFarDistributionRow] = []
    for raw_row in rows:
        far_ceiling = _maybe_float(raw_row.get("far_ceiling"))
        if far_ceiling is None:
            continue
        distribution.append(
            BenchmarkTarFarDistributionRow(
                far_ceiling=far_ceiling,
                threshold=_maybe_float(raw_row.get("threshold")),
                actual_far=_maybe_float(raw_row.get("actual_far")),
                tar=_maybe_float(raw_row.get("tar")),
                frr=_maybe_float(raw_row.get("frr")),
                tnr=_maybe_float(raw_row.get("tnr")),
                ta=_maybe_int(raw_row.get("ta")),
                fr=_maybe_int(raw_row.get("fr")),
                fa=_maybe_int(raw_row.get("fa")),
                tr=_maybe_int(raw_row.get("tr")),
                n_positive=_maybe_int(raw_row.get("n_positive")),
                n_negative=_maybe_int(raw_row.get("n_negative")),
            )
        )
    return sorted(distribution, key=lambda row: row.far_ceiling)


def _build_plain_roll_final_entries(
    source_key: str,
    source_label: str,
    source_root: Path,
) -> List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]]:
    bundle_dir = source_root / PLAIN_ROLL_FINAL_BUNDLE
    metrics_path = bundle_dir / "plain_roll_final_metrics.csv"
    thresholds_path = bundle_dir / "plain_roll_final_thresholds.csv"
    threshold_sweep_path = bundle_dir / "plain_roll_final_threshold_sweep.csv"
    tar_far_distribution_path = bundle_dir / "plain_roll_final_tar_far_distribution.csv"
    if not _is_available_artifact_file(metrics_path) or not _is_available_artifact_file(thresholds_path):
        return []

    metrics_rows = _safe_read_csv(metrics_path)
    threshold_rows_raw = _safe_read_csv(thresholds_path)
    if not metrics_rows or not threshold_rows_raw:
        return []

    manifest_path = bundle_dir / "plain_roll_final_manifest.json"
    manifest = _safe_read_json(manifest_path)
    failures_path = bundle_dir / "plain_roll_final_failures.csv"
    failures_empty = _plain_roll_final_failures_empty(failures_path)
    failure_count = _maybe_int(manifest.get("failure_count")) if manifest else None
    manifest_clean = failure_count == 0
    created_at = str(manifest.get("created_at") or "").strip() or None
    git_payload = manifest.get("git") if isinstance(manifest.get("git"), dict) else {}
    git_commit = str(git_payload.get("commit") or "").strip() or None
    is_dirty = bool(git_payload.get("is_dirty")) if git_payload else False
    source_display_label = FINAL_EVIDENCE_SOURCE_LABEL if source_key == "live" else f"{FINAL_EVIDENCE_SOURCE_LABEL} ({source_label})"

    metrics_by_key: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    for raw_row in metrics_rows:
        if str(raw_row.get("split") or "").strip() != "test":
            continue
        method = str(raw_row.get("method") or "").strip()
        dataset = str(raw_row.get("dataset") or "").strip()
        target_far = _maybe_float(raw_row.get("target_far"))
        if not method or not dataset or target_far is None:
            continue
        metrics_by_key[(dataset, method, target_far)] = raw_row

    thresholds_by_key: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    for raw_row in threshold_rows_raw:
        method = str(raw_row.get("method") or "").strip()
        dataset = str(raw_row.get("dataset") or "").strip()
        target_far = _maybe_float(raw_row.get("target_far"))
        if not method or not dataset or target_far is None:
            continue
        thresholds_by_key[(dataset, method, target_far)] = raw_row

    distribution_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for raw_row in _safe_read_csv(tar_far_distribution_path):
        if str(raw_row.get("split") or "").strip() != "test":
            continue
        method = str(raw_row.get("method") or "").strip()
        dataset = str(raw_row.get("dataset") or "").strip()
        if not method or not dataset:
            continue
        distribution_by_key.setdefault((dataset, method), []).append(raw_row)

    benchmark_methods_in_run = list(PLAIN_ROLL_FINAL_METHODS)
    methods_in_run = [
        benchmark_method_to_canonical(method)
        for method in benchmark_methods_in_run
        if benchmark_method_to_canonical(method)
    ]
    research_benchmark_methods = [
        method
        for method in benchmark_methods_in_run
        if method_research_track(method) or not method_showcase_eligible(method)
    ]
    research_methods = [
        benchmark_method_to_canonical(method)
        for method in research_benchmark_methods
        if benchmark_method_to_canonical(method)
    ]

    entries: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]] = []
    for dataset in PLAIN_ROLL_FINAL_DATASETS:
        run = _plain_roll_final_run(dataset)
        audit_ok = _plain_roll_final_pair_audit_ok(manifest, dataset, "test")
        validated = manifest_clean and failures_empty and audit_ok and not is_dirty
        status = "validated" if validated else "partial"
        validation_state = "validated" if validated else "partial"
        comparison_rows: List[ComparisonRow] = []

        for benchmark_method in sorted(PLAIN_ROLL_FINAL_METHODS, key=lambda item: PLAIN_ROLL_FINAL_DISPLAY_ORDER[item]):
            final_markdown = bundle_dir / _plain_roll_final_markdown_filename(dataset, benchmark_method)
            if not _is_available_artifact_file(final_markdown):
                continue

            method_rows = {
                target_far: metrics_by_key[(dataset, benchmark_method, target_far)]
                for target_far in (0.01, 0.005)
                if (dataset, benchmark_method, target_far) in metrics_by_key
            }
            if not method_rows:
                continue
            threshold_rows = {
                target_far: thresholds_by_key[(dataset, benchmark_method, target_far)]
                for target_far in (0.01, 0.005)
                if (dataset, benchmark_method, target_far) in thresholds_by_key
            }
            primary_row = method_rows.get(0.01) or next(iter(method_rows.values()))
            method = benchmark_method_to_canonical(benchmark_method)
            method_label = canonical_method_label_from_benchmark(benchmark_method)
            presentation_meta = _plain_roll_final_method_metadata(benchmark_method)
            showcase_exclusion_note = presentation_meta.get("showcase_exclusion_note") or _showcase_exclusion_note(method, benchmark_method)
            role_note = PLAIN_ROLL_FINAL_METHOD_ROLES.get(benchmark_method, "Final classical baseline evidence.")
            operating_points = _build_plain_roll_final_operating_points(method_rows, threshold_rows)
            tar_far_distribution = _build_plain_roll_final_tar_far_distribution(
                distribution_by_key.get((dataset, benchmark_method), [])
            )
            tar_at_far_1e_2 = next(
                (point.test_tar for point in operating_points if abs(point.target_far - 0.01) < 1e-12),
                None,
            )
            n_positive = _maybe_int(primary_row.get("n_positive"))
            n_negative = _maybe_int(primary_row.get("n_negative"))
            n_pairs = _maybe_int(primary_row.get("n_scored"))
            if n_pairs is None and n_positive is not None and n_negative is not None:
                n_pairs = n_positive + n_negative
            latency_ms = _maybe_float(primary_row.get("avg_ms_pair_reported"))
            scores_csv = _final_bundle_path_from_row(
                source_root,
                primary_row.get("scores_csv"),
                bundle_dir / f"scores_{dataset}_{benchmark_method}_test.csv",
            )
            meta_json = _final_bundle_path_from_row(
                source_root,
                primary_row.get("run_meta_json"),
                bundle_dir / f"run_{dataset}_{benchmark_method}_test.meta.json",
            )
            method_meta_json = _final_bundle_path_from_row(
                source_root,
                primary_row.get("method_meta_json"),
                bundle_dir / f"scores_{dataset}_{benchmark_method}_test.csv.meta.json",
            )
            selected_pairs = _final_bundle_path_from_row(
                source_root,
                primary_row.get("selected_pairs_csv"),
                bundle_dir / "selected_pairs" / f"pairs_{dataset}_test.csv",
            )

            artifacts = [
                _source_artifact_link(run, source_root, "summary_csv", metrics_path),
                _source_artifact_link(run, source_root, "thresholds_csv", thresholds_path),
                _source_artifact_link(run, source_root, "threshold_sweep_csv", threshold_sweep_path),
                _source_artifact_link(run, source_root, "tar_far_distribution_csv", tar_far_distribution_path),
                _source_artifact_link(run, source_root, "scores_csv", scores_csv),
                _source_artifact_link(run, source_root, "meta_json", meta_json or method_meta_json),
                _source_artifact_link(run, source_root, "roc_png", bundle_dir / f"roc_{dataset}_{benchmark_method}_test.png"),
                _source_artifact_link(run, source_root, "markdown_summary", bundle_dir / "plain_roll_final_summary.md"),
                _source_artifact_link(run, source_root, "final_markdown", final_markdown),
                _source_artifact_link(run, source_root, "run_manifest", manifest_path),
                _source_artifact_link(run, source_root, "latency_summary", bundle_dir / "plain_roll_final_latency_summary.csv"),
                _source_artifact_link(run, source_root, "positive_only_metrics", bundle_dir / "plain_roll_final_positive_only_metrics.csv"),
                _source_artifact_link(run, source_root, "negative_only_metrics", bundle_dir / "plain_roll_final_negative_only_metrics.csv"),
                _source_artifact_link(run, source_root, "failures_csv", failures_path),
            ]
            available_artifacts = _available_artifact_keys(artifacts)
            failure_text = "0 recorded failures" if failures_empty and manifest_clean else "failure report requires review"
            pair_text = (
                f"{n_positive:,} positive / {n_negative:,} negative TEST pairs"
                if n_positive is not None and n_negative is not None
                else f"{n_pairs:,} TEST pairs" if n_pairs is not None else "TEST pair count unavailable"
            )
            summary_text = (
                f"{role_note} {pair_text}. VAL calibration used 700 positive / 700 negative pairs. "
                f"{failure_text}. Positive-only and negative-only metrics are available as separate final artifacts."
            )

            provenance = BenchmarkProvenance(
                run=run,
                run_label=f"Final classical plain-vs-roll evidence ({dataset})",
                run_kind="full",
                view_mode="canonical",
                status=status,
                validation_state=validation_state,
                source_type="plain_roll_final",
                artifact_source=_plain_roll_final_markdown_filename(dataset, benchmark_method),
                methods_in_run=methods_in_run,
                benchmark_methods_in_run=benchmark_methods_in_run,
                showcase_methods_in_run=[],
                showcase_benchmark_methods_in_run=[],
                research_methods_in_run=research_methods,
                research_benchmark_methods_in_run=research_benchmark_methods,
                canonical_method=method,
                benchmark_method=benchmark_method,
                method_label=method_label,
                method_status=presentation_meta["method_status"],
                presentation_tier=presentation_meta["presentation_tier"],
                showcase_eligible=bool(presentation_meta["showcase_eligible"]),
                benchmark_default=bool(presentation_meta["benchmark_default"]),
                canonical_default=bool(presentation_meta["canonical_default"]),
                research_track=bool(presentation_meta["research_track"]),
                not_champion_candidate=bool(presentation_meta["not_champion_candidate"]),
                timestamp_utc=created_at,
                pairs_path=str(selected_pairs) if selected_pairs is not None else None,
                manifest_path=str(manifest_path),
                data_dir=str(bundle_dir),
                git_commit=git_commit,
                available_artifacts=available_artifacts,
                benchmark_source_root=source_key,
                benchmark_source_label=source_display_label,
                showcase_exclusion_note=showcase_exclusion_note,
            )

            comparison_rows.append(
                ComparisonRow(
                    dataset=dataset,
                    run=run,
                    split="test",
                    method=method,
                    benchmark_method=benchmark_method,
                    method_label=method_label,
                    method_status=presentation_meta["method_status"],
                    presentation_tier=presentation_meta["presentation_tier"],
                    showcase_eligible=bool(presentation_meta["showcase_eligible"]),
                    benchmark_default=bool(presentation_meta["benchmark_default"]),
                    canonical_default=bool(presentation_meta["canonical_default"]),
                    research_track=bool(presentation_meta["research_track"]),
                    not_champion_candidate=bool(presentation_meta["not_champion_candidate"]),
                    showcase_exclusion_note=showcase_exclusion_note,
                    auc=float(_maybe_float(primary_row.get("auc")) or 0.0),
                    eer=float(_maybe_float(primary_row.get("eer")) or 0.0),
                    n_pairs=n_pairs,
                    tar_at_far_1e_2=tar_at_far_1e_2,
                    tar_at_far_1e_3=None,
                    operating_points=operating_points,
                    tar_far_distribution=tar_far_distribution,
                    latency_ms=latency_ms,
                    latency_source="reported" if latency_ms is not None else None,
                    run_family=PLAIN_ROLL_FINAL_BUNDLE,
                    run_label=f"Final classical plain-vs-roll evidence ({dataset})",
                    run_kind="full",
                    view_mode="canonical",
                    status=status,
                    validation_state=validation_state,
                    artifact_count=len(available_artifacts),
                    available_artifacts=available_artifacts,
                    summary_text=summary_text,
                    artifacts=artifacts,
                    provenance=provenance,
                )
            )

        if not comparison_rows:
            continue

        entries.append(
            (
                BenchmarkRunInfo(
                    run=run,
                    dataset=dataset,
                    run_kind="full",
                    view_mode="canonical",
                    status=status,
                    validation_state=validation_state,
                    validated=validated,
                    recommended=False,
                    run_label=f"Final classical plain-vs-roll evidence ({dataset})",
                    artifact_count=max((row.artifact_count for row in comparison_rows), default=0),
                    summary_note=(
                        "Final pair-audited classical benchmark bundle selected for the main showcase; "
                        f"{'failures CSV is empty' if failures_empty else 'failures CSV requires review'}."
                    ),
                    methods=methods_in_run,
                    benchmark_methods=benchmark_methods_in_run,
                    showcase_methods=[],
                    showcase_benchmark_methods=[],
                    research_methods=research_methods,
                    research_benchmark_methods=research_benchmark_methods,
                    splits=["test"],
                    dataset_info=_dataset_info(dataset),
                    benchmark_source_root=source_key,
                    benchmark_source_label=source_display_label,
                ),
                comparison_rows,
            )
        )

    return entries


def _sourceafis_dataset_dpi(dataset: str) -> Optional[int]:
    if dataset == "nist_sd300b":
        return 1000
    if dataset == "nist_sd300c":
        return 2000
    return None


def _build_sourceafis_final_entries(
    source_key: str,
    source_label: str,
    source_root: Path,
) -> List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]]:
    bundle_dir = source_root / SOURCEAFIS_FINAL_BUNDLE
    metrics_path = bundle_dir / "plain_roll_final_metrics.csv"
    thresholds_path = bundle_dir / "plain_roll_final_thresholds.csv"
    threshold_sweep_path = bundle_dir / "plain_roll_final_threshold_sweep.csv"
    tar_far_distribution_path = bundle_dir / "plain_roll_final_tar_far_distribution.csv"
    if not _is_available_artifact_file(metrics_path) or not _is_available_artifact_file(thresholds_path):
        return []

    metrics_rows = _safe_read_csv(metrics_path)
    threshold_rows_raw = _safe_read_csv(thresholds_path)
    if not metrics_rows or not threshold_rows_raw:
        return []

    manifest_path = bundle_dir / "plain_roll_final_manifest.json"
    manifest = _safe_read_json(manifest_path)
    failures_path = bundle_dir / "plain_roll_final_failures.csv"
    failures_empty = _plain_roll_final_failures_empty(failures_path)
    failure_count = _maybe_int(manifest.get("failure_count")) if manifest else None
    manifest_clean = failure_count == 0
    created_at = str(manifest.get("created_at") or "").strip() or None
    git_payload = manifest.get("git") if isinstance(manifest.get("git"), dict) else {}
    git_commit = str(git_payload.get("commit") or "").strip() or None
    is_dirty = bool(git_payload.get("is_dirty")) if git_payload else False
    source_display_label = FINAL_EVIDENCE_SOURCE_LABEL if source_key == "live" else f"{FINAL_EVIDENCE_SOURCE_LABEL} ({source_label})"

    metrics_by_key: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    for raw_row in metrics_rows:
        if str(raw_row.get("split") or "").strip() != "test":
            continue
        method = str(raw_row.get("method") or "").strip()
        dataset = str(raw_row.get("dataset") or "").strip()
        target_far = _maybe_float(raw_row.get("target_far"))
        if not method or not dataset or target_far is None:
            continue
        metrics_by_key[(dataset, method, target_far)] = raw_row

    thresholds_by_key: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    for raw_row in threshold_rows_raw:
        method = str(raw_row.get("method") or "").strip()
        dataset = str(raw_row.get("dataset") or "").strip()
        target_far = _maybe_float(raw_row.get("target_far"))
        if not method or not dataset or target_far is None:
            continue
        thresholds_by_key[(dataset, method, target_far)] = raw_row

    distribution_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for raw_row in _safe_read_csv(tar_far_distribution_path):
        if str(raw_row.get("split") or "").strip() != "test":
            continue
        method = str(raw_row.get("method") or "").strip()
        dataset = str(raw_row.get("dataset") or "").strip()
        if not method or not dataset:
            continue
        distribution_by_key.setdefault((dataset, method), []).append(raw_row)

    benchmark_method = "sourceafis_open"
    method = benchmark_method_to_canonical(benchmark_method) or benchmark_method
    method_label = SOURCEAFIS_METHOD_LABEL
    presentation_meta = _method_metadata_for_final_evidence(benchmark_method, {})
    benchmark_methods_in_run = list(SOURCEAFIS_FINAL_METHODS)
    methods_in_run = [method]
    entries: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]] = []

    for dataset in PLAIN_ROLL_FINAL_DATASETS:
        run = _sourceafis_final_run(dataset)
        audit_ok = _plain_roll_final_pair_audit_ok(manifest, dataset, "test")
        validated = manifest_clean and failures_empty and audit_ok and not is_dirty
        status = "validated" if validated else "partial"
        validation_state = "validated" if validated else "partial"
        final_markdown = bundle_dir / _plain_roll_final_markdown_filename(dataset, benchmark_method)
        if not _is_available_artifact_file(final_markdown):
            continue

        method_rows = {
            target_far: metrics_by_key[(dataset, benchmark_method, target_far)]
            for target_far in (0.01, 0.005)
            if (dataset, benchmark_method, target_far) in metrics_by_key
        }
        if not method_rows:
            continue
        threshold_rows = {
            target_far: thresholds_by_key[(dataset, benchmark_method, target_far)]
            for target_far in (0.01, 0.005)
            if (dataset, benchmark_method, target_far) in thresholds_by_key
        }
        primary_row = method_rows.get(0.01) or next(iter(method_rows.values()))
        operating_points = _build_plain_roll_final_operating_points(method_rows, threshold_rows)
        tar_far_distribution = _build_plain_roll_final_tar_far_distribution(
            distribution_by_key.get((dataset, benchmark_method), [])
        )
        tar_at_far_1e_2 = next(
            (point.test_tar for point in operating_points if abs(point.target_far - 0.01) < 1e-12),
            None,
        )
        n_positive = _maybe_int(primary_row.get("n_positive"))
        n_negative = _maybe_int(primary_row.get("n_negative"))
        n_pairs = _maybe_int(primary_row.get("n_scored"))
        if n_pairs is None and n_positive is not None and n_negative is not None:
            n_pairs = n_positive + n_negative
        latency_ms = _maybe_float(primary_row.get("avg_ms_pair_reported"))
        scores_csv = _final_bundle_path_from_row(
            source_root,
            primary_row.get("scores_csv"),
            bundle_dir / "scores" / f"scores_{dataset}_{benchmark_method}_test.csv",
        )
        meta_json = _final_bundle_path_from_row(
            source_root,
            primary_row.get("run_meta_json"),
            bundle_dir / "run_meta" / f"run_{dataset}_{benchmark_method}_test.meta.json",
        )
        method_meta_json = _final_bundle_path_from_row(
            source_root,
            primary_row.get("method_meta_json"),
            bundle_dir / "scores" / f"scores_{dataset}_{benchmark_method}_test.meta.json",
        )
        selected_pairs = _final_bundle_path_from_row(
            source_root,
            primary_row.get("selected_pairs_csv"),
            bundle_dir / "selected_pairs" / f"pairs_{dataset}_test.csv",
        )

        artifacts = [
            _source_artifact_link(run, source_root, "summary_csv", metrics_path),
            _source_artifact_link(run, source_root, "thresholds_csv", thresholds_path),
            _source_artifact_link(run, source_root, "threshold_sweep_csv", threshold_sweep_path),
            _source_artifact_link(run, source_root, "tar_far_distribution_csv", tar_far_distribution_path),
            _source_artifact_link(run, source_root, "scores_csv", scores_csv),
            _source_artifact_link(run, source_root, "meta_json", meta_json or method_meta_json),
            _source_artifact_link(run, source_root, "roc_png", bundle_dir / "roc" / f"roc_{dataset}_{benchmark_method}_test.png"),
            _source_artifact_link(run, source_root, "markdown_summary", bundle_dir / "plain_roll_final_summary.md"),
            _source_artifact_link(run, source_root, "final_markdown", final_markdown),
            _source_artifact_link(run, source_root, "run_manifest", manifest_path),
            _source_artifact_link(run, source_root, "latency_summary", bundle_dir / "plain_roll_final_latency_summary.csv"),
            _source_artifact_link(run, source_root, "positive_only_metrics", bundle_dir / "plain_roll_final_positive_only_metrics.csv"),
            _source_artifact_link(run, source_root, "negative_only_metrics", bundle_dir / "plain_roll_final_negative_only_metrics.csv"),
            _source_artifact_link(run, source_root, "failures_csv", failures_path),
        ]
        available_artifacts = _available_artifact_keys(artifacts)
        failure_text = "0 recorded failures" if failures_empty and manifest_clean else "failure report requires review"
        pair_text = (
            f"{n_positive:,} positive / {n_negative:,} negative TEST pairs"
            if n_positive is not None and n_negative is not None
            else f"{n_pairs:,} TEST pairs" if n_pairs is not None else "TEST pair count unavailable"
        )
        summary_text = (
            f"SourceAFIS champion evidence under the same audited plain-vs-roll selected pairs. {pair_text}. "
            f"VAL calibration used 700 positive / 700 negative pairs. {failure_text}. "
            "Expert TAR/FAR distribution plus positive-only and negative-only metrics are available as final artifacts."
        )
        showcase_exclusion_note = None

        provenance = BenchmarkProvenance(
            run=run,
            run_label=f"Final SourceAFIS plain-vs-roll evidence ({dataset})",
            run_kind="full",
            view_mode="canonical",
            status=status,
            validation_state=validation_state,
            source_type="plain_roll_final_sourceafis",
            artifact_source=_plain_roll_final_markdown_filename(dataset, benchmark_method),
            methods_in_run=methods_in_run,
            benchmark_methods_in_run=benchmark_methods_in_run,
            showcase_methods_in_run=methods_in_run,
            showcase_benchmark_methods_in_run=benchmark_methods_in_run,
            research_methods_in_run=[],
            research_benchmark_methods_in_run=[],
            canonical_method=method,
            benchmark_method=benchmark_method,
            method_label=method_label,
            method_status=presentation_meta["method_status"],
            presentation_tier=presentation_meta["presentation_tier"],
            showcase_eligible=bool(presentation_meta["showcase_eligible"]),
            benchmark_default=bool(presentation_meta["benchmark_default"]),
            canonical_default=bool(presentation_meta["canonical_default"]),
            research_track=bool(presentation_meta["research_track"]),
            not_champion_candidate=bool(presentation_meta["not_champion_candidate"]),
            timestamp_utc=created_at,
            pairs_path=str(selected_pairs) if selected_pairs is not None else None,
            manifest_path=str(manifest_path),
            data_dir=str(bundle_dir),
            git_commit=git_commit,
            available_artifacts=available_artifacts,
            benchmark_source_root=source_key,
            benchmark_source_label=source_display_label,
            showcase_exclusion_note=showcase_exclusion_note,
        )

        row = ComparisonRow(
            dataset=dataset,
            run=run,
            split="test",
            method=method,
            benchmark_method=benchmark_method,
            method_label=method_label,
            method_status=presentation_meta["method_status"],
            presentation_tier=presentation_meta["presentation_tier"],
            showcase_eligible=bool(presentation_meta["showcase_eligible"]),
            benchmark_default=bool(presentation_meta["benchmark_default"]),
            canonical_default=bool(presentation_meta["canonical_default"]),
            research_track=bool(presentation_meta["research_track"]),
            not_champion_candidate=bool(presentation_meta["not_champion_candidate"]),
            showcase_exclusion_note=showcase_exclusion_note,
            auc=float(_maybe_float(primary_row.get("auc")) or 0.0),
            eer=float(_maybe_float(primary_row.get("eer")) or 0.0),
            n_pairs=n_pairs,
            dpi=_sourceafis_dataset_dpi(dataset),
            tar_at_far_1e_2=tar_at_far_1e_2,
            tar_at_far_1e_3=None,
            operating_points=operating_points,
            tar_far_distribution=tar_far_distribution,
            latency_ms=latency_ms,
            latency_source="reported" if latency_ms is not None else None,
            run_family=SOURCEAFIS_FINAL_BUNDLE,
            run_label=f"Final SourceAFIS plain-vs-roll evidence ({dataset})",
            run_kind="full",
            view_mode="canonical",
            status=status,
            validation_state=validation_state,
            artifact_count=len(available_artifacts),
            available_artifacts=available_artifacts,
            summary_text=summary_text,
            artifacts=artifacts,
            provenance=provenance,
        )

        entries.append(
            (
                BenchmarkRunInfo(
                    run=run,
                    dataset=dataset,
                    run_kind="full",
                    view_mode="canonical",
                    status=status,
                    validation_state=validation_state,
                    validated=validated,
                    recommended=validated,
                    run_label=f"Final SourceAFIS plain-vs-roll evidence ({dataset})",
                    artifact_count=row.artifact_count,
                    summary_note=(
                        "Final pair-audited SourceAFIS benchmark bundle selected for the main showcase; "
                        f"{'failures CSV is empty' if failures_empty else 'failures CSV requires review'}."
                    ),
                    methods=methods_in_run,
                    benchmark_methods=benchmark_methods_in_run,
                    showcase_methods=methods_in_run,
                    showcase_benchmark_methods=benchmark_methods_in_run,
                    research_methods=[],
                    research_benchmark_methods=[],
                    splits=["test"],
                    dataset_info=_dataset_info(dataset),
                    benchmark_source_root=source_key,
                    benchmark_source_label=source_display_label,
                ),
                [row],
            )
        )

    return entries


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


def _build_final_evidence_entries(source_key: str, source_label: str, source_root: Path) -> List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]]:
    entries: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]] = []
    if not source_root.exists():
        return entries
    sourceafis_bundle_available = _is_available_artifact_file(
        source_root / SOURCEAFIS_FINAL_BUNDLE / "plain_roll_final_metrics.csv"
    )

    for spec in CURATED_FINAL_EVIDENCE:
        filename = str(spec["filename"])
        markdown_path = source_root / filename
        if not _is_available_artifact_file(markdown_path):
            continue

        benchmark_method = str(spec["method"])
        if benchmark_method == "sourceafis_open" and sourceafis_bundle_available:
            continue
        method = benchmark_method_to_canonical(benchmark_method)
        method_label = _method_label_for_final_evidence(benchmark_method, spec)
        presentation_meta = _method_metadata_for_final_evidence(benchmark_method, spec)
        if spec.get("note") and benchmark_method != "sourceafis_open":
            presentation_meta = {
                **presentation_meta,
                "showcase_exclusion_note": str(spec["note"]),
            }
        showcase_exclusion_note = presentation_meta.get("showcase_exclusion_note") or _showcase_exclusion_note(method, benchmark_method)
        run = str(spec["run"])
        run_label = str(spec["run_label"])
        dataset = str(spec["dataset"])
        split = str(spec.get("split") or "test")
        operating_points = _curated_operating_points(spec)
        tar_at_far_1e_2 = next(
            (point.test_tar for point in operating_points if abs(point.target_far - 0.01) < 1e-12),
            None,
        )
        artifacts = [
            _loose_artifact_link(run, source_root, "final_markdown", filename),
        ]
        available_artifacts = _available_artifact_keys(artifacts)
        source_display_label = FINAL_EVIDENCE_SOURCE_LABEL if source_key == "live" else f"{FINAL_EVIDENCE_SOURCE_LABEL} ({source_label})"
        n_pairs = _maybe_int(spec.get("n_pairs"))
        latency_ms = _maybe_float(spec.get("latency_ms"))
        dpi = _maybe_int(spec.get("dpi"))
        summary_text = (
            f"{spec['summary_prefix']}. {n_pairs:,} scored TEST pairs. "
            f"{spec.get('note') or 'Final markdown evidence is preserved as the source artifact.'}"
        )

        provenance = BenchmarkProvenance(
            run=run,
            run_label=run_label,
            run_kind="full",
            view_mode="canonical",
            status="validated",
            validation_state="validated",
            source_type="final_markdown",
            artifact_source=filename,
            methods_in_run=[method],
            benchmark_methods_in_run=[benchmark_method],
            showcase_methods_in_run=[] if presentation_meta["research_track"] else [method],
            showcase_benchmark_methods_in_run=[] if presentation_meta["research_track"] else [benchmark_method],
            research_methods_in_run=[method] if presentation_meta["research_track"] else [],
            research_benchmark_methods_in_run=[benchmark_method] if presentation_meta["research_track"] else [],
            canonical_method=method,
            benchmark_method=benchmark_method,
            method_label=method_label,
            method_status=presentation_meta["method_status"],
            presentation_tier=presentation_meta["presentation_tier"],
            showcase_eligible=bool(presentation_meta["showcase_eligible"]),
            benchmark_default=bool(presentation_meta["benchmark_default"]),
            canonical_default=bool(presentation_meta["canonical_default"]),
            research_track=bool(presentation_meta["research_track"]),
            not_champion_candidate=bool(presentation_meta["not_champion_candidate"]),
            timestamp_utc=str(spec.get("timestamp_utc") or "").strip() or None,
            manifest_path=str(spec.get("manifest_path") or "").strip() or None,
            data_dir=str(spec.get("data_dir") or "").strip() or None,
            git_commit=str(spec.get("git_commit") or "").strip() or None,
            available_artifacts=available_artifacts,
            benchmark_source_root=source_key,
            benchmark_source_label=source_display_label,
            showcase_exclusion_note=showcase_exclusion_note,
        )

        row = ComparisonRow(
            dataset=dataset,
            run=run,
            split=split,
            method=method,
            benchmark_method=benchmark_method,
            method_label=method_label,
            method_status=presentation_meta["method_status"],
            presentation_tier=presentation_meta["presentation_tier"],
            showcase_eligible=bool(presentation_meta["showcase_eligible"]),
            benchmark_default=bool(presentation_meta["benchmark_default"]),
            canonical_default=bool(presentation_meta["canonical_default"]),
            research_track=bool(presentation_meta["research_track"]),
            not_champion_candidate=bool(presentation_meta["not_champion_candidate"]),
            showcase_exclusion_note=showcase_exclusion_note,
            auc=float(spec["auc"]),
            eer=float(spec["eer"]),
            n_pairs=n_pairs,
            dpi=dpi,
            tar_at_far_1e_2=tar_at_far_1e_2,
            tar_at_far_1e_3=None,
            operating_points=operating_points,
            latency_ms=latency_ms,
            latency_source="reported" if latency_ms is not None else None,
            run_family=run,
            run_label=run_label,
            run_kind="full",
            view_mode="canonical",
            status="validated",
            validation_state="validated",
            artifact_count=len(available_artifacts),
            available_artifacts=available_artifacts,
            summary_text=summary_text,
            artifacts=artifacts,
            provenance=provenance,
        )

        run_info = BenchmarkRunInfo(
            run=run,
            dataset=dataset,
            run_kind="full",
            view_mode="canonical",
            status="validated",
            validation_state="validated",
            validated=True,
            recommended=bool(spec.get("recommended")),
            run_label=run_label,
            artifact_count=len(available_artifacts),
            summary_note="Final curated markdown evidence selected for the main showcase.",
            methods=[method],
            benchmark_methods=[benchmark_method],
            showcase_methods=[] if presentation_meta["research_track"] else [method],
            showcase_benchmark_methods=[] if presentation_meta["research_track"] else [benchmark_method],
            research_methods=[method] if presentation_meta["research_track"] else [],
            research_benchmark_methods=[benchmark_method] if presentation_meta["research_track"] else [],
            splits=[split],
            dataset_info=_dataset_info(dataset),
            benchmark_source_root=source_key,
            benchmark_source_label=source_display_label,
        )
        entries.append((run_info, [row]))

    return entries


def _scan_runs(
    root: Path = BENCH_ROOT,
    reference_root: Path | None = None,
) -> List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]]:
    catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]] = []
    seen_runs: set[str] = set()

    for source_key, source_label, source_root in _benchmark_sources(root, reference_root):
        if not source_root.exists():
            continue

        for run_dir in sorted(source_root.iterdir(), key=lambda item: item.name):
            if not run_dir.is_dir():
                continue
            if excluded_run_name(run_dir.name):
                continue
            if run_dir.name in seen_runs:
                continue
            seen_runs.add(run_dir.name)

            summary_csv = run_dir / "results_summary.csv"
            if not _is_available_artifact_file(summary_csv):
                continue

            try:
                raw_rows = _read_summary_csv(summary_csv)
            except OSError:
                continue

            manifest_payload = _safe_read_json(run_dir / "run_manifest.json")
            dataset = _infer_dataset(run_dir.name, manifest_payload, raw_rows)
            if not is_showcase_dataset(dataset):
                continue

            validation_ok = (run_dir / "validation.ok").exists()
            run_kind = infer_run_kind(run_dir.name)
            view_mode = infer_view_mode(run_dir.name, validated=validation_ok, source=source_key)
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
            showcase_benchmark_methods_in_run = sorted(
                {
                    method
                    for method in methods_in_run
                    if method_showcase_eligible(method) and not method_research_track(method)
                },
                key=benchmark_method_sort_key,
            )
            showcase_methods_in_run = sorted(
                {
                    benchmark_method_to_canonical(method)
                    for method in showcase_benchmark_methods_in_run
                    if benchmark_method_to_canonical(method)
                },
                key=canonical_method_sort_key,
            )
            research_benchmark_methods_in_run = sorted(
                {
                    method
                    for method in methods_in_run
                    if method_research_track(method) or not method_showcase_eligible(method)
                },
                key=benchmark_method_sort_key,
            )
            research_methods_in_run = sorted(
                {
                    benchmark_method_to_canonical(method)
                    for method in research_benchmark_methods_in_run
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
                presentation_meta = method_presentation_metadata(benchmark_method)
                showcase_exclusion_note = _showcase_exclusion_note(method, benchmark_method)

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
                operating_points = _legacy_operating_points(raw_row)

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
                validation_state = _validation_state_for_run(view_mode, validated=validation_ok, partial=partial)
                status = _status_for_run(view_mode, validated=validation_ok, partial=partial)
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
                    showcase_methods_in_run=showcase_methods_in_run,
                    showcase_benchmark_methods_in_run=showcase_benchmark_methods_in_run,
                    research_methods_in_run=research_methods_in_run,
                    research_benchmark_methods_in_run=research_benchmark_methods_in_run,
                    canonical_method=method,
                    benchmark_method=benchmark_method,
                    method_label=method_label,
                    method_status=presentation_meta["method_status"],
                    presentation_tier=presentation_meta["presentation_tier"],
                    showcase_eligible=bool(presentation_meta["showcase_eligible"]),
                    benchmark_default=bool(presentation_meta["benchmark_default"]),
                    canonical_default=bool(presentation_meta["canonical_default"]),
                    research_track=bool(presentation_meta["research_track"]),
                    not_champion_candidate=bool(presentation_meta["not_champion_candidate"]),
                    timestamp_utc=str(raw_row.get("timestamp_utc", "")).strip() or manifest_payload.get("timestamp_utc"),
                    limit=limit,
                    pairs_path=str(method_meta.get("pairs_path") or config_json.get("pairs_path") or "").strip() or None,
                    manifest_path=str(method_meta.get("manifest_path") or config_json.get("manifest_path") or "").strip() or None,
                    data_dir=str(method_meta.get("resolved_data_dir") or config_json.get("resolved_data_dir") or manifest_payload.get("data_dir") or "").strip() or None,
                    git_commit=str(manifest_payload.get("git", {}).get("commit") or "").strip() or None,
                    available_artifacts=available_artifacts,
                    benchmark_source_root=source_key,
                    benchmark_source_label=source_label,
                    showcase_exclusion_note=showcase_exclusion_note,
                )

                comparison_rows.append(
                    ComparisonRow(
                        dataset=dataset,
                        run=run_dir.name,
                        split=split,
                        method=method,
                        benchmark_method=benchmark_method,
                        method_label=method_label,
                        method_status=presentation_meta["method_status"],
                        presentation_tier=presentation_meta["presentation_tier"],
                        showcase_eligible=bool(presentation_meta["showcase_eligible"]),
                        benchmark_default=bool(presentation_meta["benchmark_default"]),
                        canonical_default=bool(presentation_meta["canonical_default"]),
                        research_track=bool(presentation_meta["research_track"]),
                        not_champion_candidate=bool(presentation_meta["not_champion_candidate"]),
                        showcase_exclusion_note=showcase_exclusion_note,
                        auc=auc,
                        eer=eer,
                        n_pairs=_maybe_int(raw_row.get("n_pairs")),
                        tar_at_far_1e_2=_maybe_float(raw_row.get("tar_at_far_1e_2")),
                        tar_at_far_1e_3=_maybe_float(raw_row.get("tar_at_far_1e_3")),
                        operating_points=operating_points,
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

            trusted_curated_reference = (
                source_key == "reference"
                and run_dir.name in CURATED_REFERENCE_RUNS
                and is_showcase_dataset(dataset)
            )
            effective_validated = validation_ok or trusted_curated_reference
            for row in comparison_rows:
                partial = row.artifact_count < 3
                row.validation_state = _validation_state_for_run(view_mode, validated=effective_validated, partial=partial)
                row.status = _status_for_run(view_mode, validated=effective_validated, partial=partial)
                row.summary_text = _row_summary_text(
                    view_mode=view_mode,
                    validation_state=row.validation_state,
                    run_label=row.run_label or format_run_label(row.run),
                    n_pairs=row.n_pairs,
                    latency_source=row.latency_source,
                )
                if row.provenance is not None:
                    row.provenance.status = row.status
                    row.provenance.validation_state = row.validation_state

            methods_sorted = sorted(methods, key=canonical_method_sort_key)
            benchmark_methods_sorted = sorted(benchmark_methods, key=benchmark_method_sort_key)
            showcase_methods_sorted = [
                method for method in methods_sorted if method_showcase_eligible(method) and not method_research_track(method)
            ]
            showcase_benchmark_methods_sorted = [
                method for method in benchmark_methods_sorted if method_showcase_eligible(method) and not method_research_track(method)
            ]
            research_methods_sorted = [
                method for method in methods_sorted if method_research_track(method) or not method_showcase_eligible(method)
            ]
            research_benchmark_methods_sorted = [
                method for method in benchmark_methods_sorted if method_research_track(method) or not method_showcase_eligible(method)
            ]
            splits_sorted = sorted(splits, key=lambda item: SPLIT_ORDER.get(item, 99))
            partial_run = any(row.status == "partial" for row in comparison_rows)
            run_info = BenchmarkRunInfo(
                run=run_dir.name,
                dataset=dataset,
                run_kind=run_kind,
                view_mode=view_mode,
                status=_status_for_run(view_mode, validated=effective_validated, partial=partial_run),
                validation_state=_validation_state_for_run(view_mode, validated=effective_validated, partial=partial_run),
                validated=effective_validated,
                recommended=run_dir.name in CANONICAL_FULL_RUNS and effective_validated and not partial_run,
                run_label=format_run_label(run_dir.name),
                artifact_count=max((row.artifact_count for row in comparison_rows), default=0),
                summary_note=_run_summary_note(view_mode, validated=effective_validated, limit=limit),
                methods=methods_sorted,
                benchmark_methods=benchmark_methods_sorted,
                showcase_methods=showcase_methods_sorted,
                showcase_benchmark_methods=showcase_benchmark_methods_sorted,
                research_methods=research_methods_sorted,
                research_benchmark_methods=research_benchmark_methods_sorted,
                splits=splits_sorted,
                dataset_info=_dataset_info(dataset),
                benchmark_source_root=source_key,
                benchmark_source_label=source_label,
            )
            catalog.append((run_info, comparison_rows))

        for run_info, comparison_rows in _build_sourceafis_final_entries(source_key, source_label, source_root):
            if run_info.run in seen_runs:
                continue
            seen_runs.add(run_info.run)
            catalog.append((run_info, comparison_rows))

        for run_info, comparison_rows in _build_plain_roll_final_entries(source_key, source_label, source_root):
            if run_info.run in seen_runs:
                continue
            seen_runs.add(run_info.run)
            catalog.append((run_info, comparison_rows))

        for run_info, comparison_rows in _build_final_evidence_entries(source_key, source_label, source_root):
            if run_info.run in seen_runs:
                continue
            seen_runs.add(run_info.run)
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
    rows = _filter_preferred_showcase_rows(rows, view_mode)
    _rank_rows(rows)
    return rows


def _row_has_showcase_evidence(row: ComparisonRow) -> bool:
    available = set(row.available_artifacts)
    if "final_markdown" in available:
        return True
    return "summary_csv" in available and bool(available & SHOWCASE_SECONDARY_EVIDENCE_KEYS)


def _best_entries_from_rows(rows: List[ComparisonRow]) -> List[BestMethodEntry]:
    best_entries: List[BestMethodEntry] = []
    candidate_rows = _champion_candidate_rows(rows)
    if not candidate_rows:
        return best_entries

    metrics = (
        ("best_auc", lambda row: row.auc_rank if row.auc_rank is not None else 999),
        ("best_eer", lambda row: row.eer_rank if row.eer_rank is not None else 999),
        ("best_latency", lambda row: row.latency_rank if row.latency_rank is not None else 999),
    )

    for metric, rank_accessor in metrics:
        sorted_rows = sorted(
            candidate_rows,
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
                method_status=winner.method_status,
                presentation_tier=winner.presentation_tier,
                showcase_eligible=winner.showcase_eligible,
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


def _available_splits_for_view(
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


def _dataset_keys_for_view(
    catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]],
    view_mode: str,
) -> List[str]:
    datasets = {
        run_info.dataset
        for run_info, run_rows in catalog
        if run_info.view_mode == view_mode and run_info.dataset and run_rows
    }
    return sorted(datasets, key=lambda item: DATASET_ORDER.get(item, 99))


def _all_dataset_keys(catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]]) -> List[str]:
    datasets = {run_info.dataset for run_info, run_rows in catalog if run_info.dataset and run_rows}
    return sorted(datasets, key=lambda item: DATASET_ORDER.get(item, 99))


def _available_view_modes(catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]]) -> List[str]:
    modes = {run_info.view_mode for run_info, run_rows in catalog if run_rows}
    return [
        view_mode
        for view_mode in VIEW_MODES
        if view_mode in modes
    ]


def _resolve_dataset(
    catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]],
    dataset: Optional[str],
    view_mode: str,
) -> str:
    available = _dataset_keys_for_view(catalog, view_mode)
    if dataset in available:
        return str(dataset)
    if available:
        return available[0]
    return SHOWCASE_DATASETS[0]


def _resolve_split(
    catalog: List[Tuple[BenchmarkRunInfo, List[ComparisonRow]]],
    dataset: str,
    split: Optional[str],
    view_mode: str,
) -> str:
    available = _available_splits_for_view(catalog, dataset=dataset, view_mode=view_mode)
    if split == "all" and available:
        return "all"
    if split and split in available:
        return split
    default_split = default_split_for_view(available, view_mode)
    return default_split or ("val" if view_mode == "smoke" else "test")


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
        "smoke": "Showing smoke benchmark regression anchors from curated smoke runs.",
        "archive": "Showing archived or non-primary benchmark rows for provenance review.",
    }.get(view_mode, "Showing curated full benchmark results from validated showcase runs.")
    if rows:
        return base
    return f"No {view_mode} benchmark results are available for the current selection."


def load_benchmark_runs(
    root: Path = BENCH_ROOT,
    reference_root: Path | None = None,
) -> BenchmarkRunsResponse:
    catalog = _scan_runs(root, reference_root=reference_root)
    items = [item for item, _ in catalog]
    canonical_datasets = _dataset_keys_for_view(catalog, SHOWCASE_VIEW_MODE)

    default_run = next((item.run for item in items if item.recommended), None)
    if default_run is None and items:
        default_run = items[0].run
    default_dataset = None
    default_split = None

    if canonical_datasets:
        default_dataset = canonical_datasets[0]
        default_split = _resolve_split(catalog, default_dataset, None, SHOWCASE_VIEW_MODE)
    elif items:
        default_dataset = _all_dataset_keys(catalog)[0] if _all_dataset_keys(catalog) else items[0].dataset
        if default_dataset:
            default_mode = next((item.view_mode for item in items if item.dataset == default_dataset), SHOWCASE_VIEW_MODE)
            default_split = _resolve_split(catalog, default_dataset, None, default_mode)

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
    reference_root: Path | None = None,
) -> BenchmarkSummaryResponse:
    normalized_view_mode = _normalize_view_mode(view_mode)
    catalog = _scan_runs(root, reference_root=reference_root)
    available_dataset_keys = _dataset_keys_for_view(catalog, normalized_view_mode)
    resolved_dataset = _resolve_dataset(catalog, dataset, normalized_view_mode)
    available_split_keys = _available_splits_for_view(catalog, dataset=resolved_dataset, view_mode=normalized_view_mode)
    resolved_split = _resolve_split(catalog, resolved_dataset, split, normalized_view_mode)
    rows = _selection_rows(catalog, dataset=resolved_dataset, split=resolved_split, view_mode=normalized_view_mode)
    validated_by_run = {run_info.run: run_info.validated for run_info, _ in catalog}
    current_runs = sorted(
        {row.run for row in rows},
        key=lambda run: run_sort_key(run, validated=validated_by_run.get(run, False)),
    )

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
        available_view_modes=[info for key in _available_view_modes(catalog) if (info := _view_info(key)) is not None],
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
    reference_root: Path | None = None,
) -> ComparisonResponse:
    normalized_view_mode = _normalize_view_mode(view_mode)
    normalized_sort_mode = sort_mode if sort_mode in SORT_MODES else "best_accuracy"
    catalog = _scan_runs(root, reference_root=reference_root)
    resolved_dataset = _resolve_dataset(catalog, dataset, normalized_view_mode)
    resolved_split = _resolve_split(catalog, resolved_dataset, split, normalized_view_mode)
    rows = _selection_rows(catalog, dataset=resolved_dataset, split=resolved_split, view_mode=normalized_view_mode)
    rows = _sort_rows(rows, normalized_sort_mode)

    dataset_info = {
        dataset_key: info
        for dataset_key in _dataset_keys_for_view(catalog, normalized_view_mode)
        if (info := _dataset_info(dataset_key)) is not None
    }
    split_keys = _available_splits_for_view(catalog, dataset=resolved_dataset, view_mode=normalized_view_mode)

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
    reference_root: Path | None = None,
) -> BestMethodsResponse:
    normalized_view_mode = _normalize_view_mode(view_mode)
    catalog = _scan_runs(root, reference_root=reference_root)
    resolved_dataset = _resolve_dataset(catalog, dataset, normalized_view_mode)
    resolved_split = _resolve_split(catalog, resolved_dataset, split, normalized_view_mode)
    rows = _selection_rows(catalog, dataset=resolved_dataset, split=resolved_split, view_mode=normalized_view_mode)
    best_entries = _best_entries_from_rows(rows)

    return BestMethodsResponse(
        dataset=resolved_dataset,
        split=resolved_split,
        view_mode=normalized_view_mode,
        entries=best_entries,
    )


def resolve_benchmark_artifact(
    run: str,
    filename: str,
    root: Path = BENCH_ROOT,
    reference_root: Path | None = None,
) -> Path:
    if not run or not filename:
        raise FileNotFoundError("Benchmark artifact request is missing run or filename.")

    catalog = _scan_runs(root, reference_root=reference_root)
    run_entry = next((item for item in catalog if item[0].run == run), None)
    if run_entry is None:
        raise FileNotFoundError(f"Unknown benchmark run: {run}")
    run_info, run_rows = run_entry

    source_roots = {
        source_key: source_root
        for source_key, _, source_root in _benchmark_sources(root, reference_root)
    }
    source_key = run_info.benchmark_source_root or "live"
    source_root = source_roots.get(source_key, root)
    run_dir = source_root / run
    if not run_dir.exists():
        requested = str(filename).replace("\\", "/").lstrip("/")
        artifact_urls = {
            artifact.url
            for row in run_rows
            for artifact in row.artifacts
            if artifact.available and artifact.url
        }
        expected_url = _artifact_url(run, requested)
        if expected_url not in artifact_urls:
            raise FileNotFoundError(f"Unknown benchmark artifact for final evidence run: {run}/{filename}")
        target = (source_root / requested).resolve()
        if not path_is_under(source_root, target):
            raise FileNotFoundError("Benchmark artifact path escaped the benchmark evidence directory.")
        if not _is_available_artifact_file(target):
            raise FileNotFoundError(f"Missing benchmark artifact: {target}")
        return target

    target = (run_dir / filename).resolve()
    if not path_is_under(run_dir, target):
        raise FileNotFoundError("Benchmark artifact path escaped the run directory.")
    if not _is_available_artifact_file(target):
        raise FileNotFoundError(f"Missing benchmark artifact: {target}")
    return target
