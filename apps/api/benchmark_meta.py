from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from apps.api.method_registry import MethodRegistryError, load_api_method_registry

DATASET_ORDER = {
    "nist_sd300b": 0,
    "nist_sd300c": 1,
    "polyu_cross": 2,
    "polyu_3d": 3,
    "unsw_2d3d": 4,
    "l3_sf_v2": 5,
}

SHOWCASE_DATASETS = [
    "nist_sd300b",
    "nist_sd300c",
    "polyu_cross",
]

SPLIT_ORDER = {"val": 0, "test": 1, "train": 2}

VIEW_MODE_ORDER = {"canonical": 0, "smoke": 1, "archive": 2}

DATASET_INFO: Dict[str, Dict[str, str]] = {
    "nist_sd300b": {
        "label": "NIST SD300b",
        "summary": "Rolled versus plain legacy benchmark at 1000 ppi, curated as the primary legacy-quality showcase.",
    },
    "nist_sd300c": {
        "label": "NIST SD300c",
        "summary": "Parallel NIST benchmark at 2000 ppi, used to show the same pipeline under a higher-resolution acquisition regime.",
    },
    "polyu_cross": {
        "label": "PolyU Cross",
        "summary": "Cross-modality evaluation between contactless and contact-based fingerprints, curated for transfer and trade-off storytelling.",
    },
    "polyu_3d": {
        "label": "PolyU 3D",
        "summary": "Research-only dataset reserved for a later phase and intentionally hidden from the main showcase.",
    },
    "unsw_2d3d": {
        "label": "UNSW 2D/3D",
        "summary": "Research-only dataset reserved for a later phase and intentionally hidden from the main showcase.",
    },
    "l3_sf_v2": {
        "label": "L3-SF v2",
        "summary": "Synthetic placeholder dataset reserved for archive or advanced analysis only.",
    },
}

SPLIT_INFO: Dict[str, Dict[str, str]] = {
    "all": {
        "label": "All available splits",
        "summary": "Aggregates every available split for the selected dataset.",
    },
    "val": {
        "label": "Validation",
        "summary": "Validation split used to compare methods before final reporting.",
    },
    "test": {
        "label": "Test",
        "summary": "Locked evaluation split used for final reporting and showcase-ready comparisons.",
    },
    "train": {
        "label": "Train",
        "summary": "Training split. Kept out of the main showcase unless explicitly requested later.",
    },
}

VIEW_MODE_INFO: Dict[str, Dict[str, str]] = {
    "canonical": {
        "label": "Canonical",
        "summary": "Validated curated benchmark families shown in the main showcase by default.",
    },
    "smoke": {
        "label": "Smoke",
        "summary": "Limited-scope regression anchors kept separate from the main benchmark story.",
    },
    "archive": {
        "label": "Archive",
        "summary": "Older or non-primary runs that stay hidden until the user opts in.",
    },
}

BEST_ORDER = {"best_auc": 0, "best_eer": 1, "best_latency": 2}

CANONICAL_FULL_RUNS = [
    "full_nist_sd300b_h6",
    "full_nist_sd300c_h6",
    "full_polyu_cross_h5",
]

CANONICAL_SMOKE_RUNS = [
    "smoke_nist_sd300b_h6",
    "smoke_nist_sd300c_h6",
    "smoke_polyu_cross_h5",
]

PREFERRED_RUN_ORDER = CANONICAL_FULL_RUNS + CANONICAL_SMOKE_RUNS

_CAPTURE_ALIASES = {
    "plain": "plain",
    "rolled": "roll",
    "roll": "roll",
    "contactless": "contactless",
    "contact-less": "contactless",
    "contact_less": "contactless",
    "contactbased": "contact_based",
    "contact-based": "contact_based",
    "contact_based": "contact_based",
}

BENCHMARK_METHOD_SEMANTICS_EPOCHS = {
    "harris": "harris_runtime_aligned_v1",
    "sift": "sift_runtime_aligned_v1",
}


def _normalize_method_name(raw: Any) -> Optional[str]:
    text = str(raw or "").strip().lower()
    return text or None


def _parse_benchmark_config_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)

    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def expected_method_semantics_epoch(method: Any) -> Optional[str]:
    normalized = _normalize_method_name(method)
    if not normalized:
        return None
    return BENCHMARK_METHOD_SEMANTICS_EPOCHS.get(normalized)


def benchmark_row_semantics_epoch(method: Any, payload: Any = None) -> Optional[str]:
    expected = expected_method_semantics_epoch(method)
    if expected is None:
        return None

    if isinstance(payload, Mapping):
        direct_epoch = str(payload.get("method_semantics_epoch") or "").strip()
        if direct_epoch:
            return direct_epoch
        config_payload = payload.get("config_json")
    else:
        config_payload = payload

    config = _parse_benchmark_config_payload(config_payload)
    epoch = str(config.get("method_semantics_epoch") or "").strip()
    return epoch or None


def benchmark_row_is_current(method: Any, payload: Any = None) -> bool:
    expected = expected_method_semantics_epoch(method)
    if expected is None:
        return True
    return benchmark_row_semantics_epoch(method, payload) == expected


def benchmark_row_is_legacy(method: Any, payload: Any = None) -> bool:
    expected = expected_method_semantics_epoch(method)
    if expected is None:
        return False
    return benchmark_row_semantics_epoch(method, payload) != expected


@lru_cache(maxsize=1)
def _method_order_lookup() -> Dict[str, int]:
    registry = load_api_method_registry()
    order: Dict[str, int] = {}

    for index, definition in enumerate(registry.list_methods()):
        lookup_names = (
            definition.canonical_api_name,
            definition.benchmark_name,
            *definition.accepted_aliases,
        )
        for lookup_name in lookup_names:
            key = _normalize_method_name(lookup_name)
            if key and key not in order:
                order[key] = index

    next_index = len(order)
    for benchmark_name in registry.benchmark_runtime_namespace:
        key = _normalize_method_name(benchmark_name)
        if key and key not in order:
            order[key] = next_index
            next_index += 1
    return order


METHOD_ORDER = _method_order_lookup()


def _resolve_registry_method(method: Any):
    normalized = _normalize_method_name(method)
    if not normalized:
        return None
    registry = load_api_method_registry()
    try:
        return registry.resolve(normalized)
    except MethodRegistryError:
        return None


def benchmark_method_sort_key(raw_benchmark_method: Any) -> tuple[int, str]:
    normalized = _normalize_method_name(raw_benchmark_method)
    if not normalized:
        return (999, "")
    return (METHOD_ORDER.get(normalized, 999), normalized)


def canonical_method_sort_key(method: Any) -> tuple[int, str]:
    normalized = benchmark_method_to_canonical(method)
    if not normalized:
        return (999, "")
    return (METHOD_ORDER.get(normalized, 999), normalized)


def benchmark_method_to_canonical(method: Any) -> str:
    normalized = _normalize_method_name(method)
    if not normalized:
        return ""

    resolved = _resolve_registry_method(normalized)
    if resolved is not None:
        return resolved.canonical_api_name

    registry = load_api_method_registry()
    canonical_name = registry.canonical_name_from_benchmark(normalized)
    return canonical_name or normalized


def canonical_method_label(method: Any) -> str:
    normalized = benchmark_method_to_canonical(method)
    if not normalized:
        return ""

    resolved = _resolve_registry_method(normalized)
    if resolved is not None:
        return resolved.ui_label

    registry = load_api_method_registry()
    definition = registry.definition_from_benchmark(normalized)
    if definition is not None:
        return definition.ui_label
    return normalized


def canonical_method_label_from_benchmark(raw_benchmark_method: Any) -> str:
    normalized = _normalize_method_name(raw_benchmark_method)
    if not normalized:
        return ""

    resolved = _resolve_registry_method(normalized)
    if resolved is not None:
        return resolved.ui_label

    registry = load_api_method_registry()
    definition = registry.definition_from_benchmark(normalized)
    if definition is not None:
        return definition.ui_label
    return benchmark_method_to_canonical(normalized)


def normalize_benchmark_context(
    context: Optional[Mapping[str, Any]],
    *,
    split: Optional[str] = None,
) -> Dict[str, Any]:
    normalized = dict(context or {})

    run = str(normalized.get("benchmark_run") or normalized.get("run") or "").strip() or None
    if run is not None:
        normalized["benchmark_run"] = run
        normalized["run"] = run

    source = str(normalized.get("source") or "").strip()
    legacy_method = _normalize_method_name(normalized.get("method"))
    raw_benchmark_method = _normalize_method_name(normalized.get("benchmark_method"))
    canonical_method = _normalize_method_name(normalized.get("canonical_method"))
    benchmark_origin = bool(run) or source.startswith("benchmark")

    if raw_benchmark_method is None and legacy_method and benchmark_origin:
        raw_benchmark_method = legacy_method

    if canonical_method is None:
        candidate = raw_benchmark_method or legacy_method
        if candidate:
            canonical_method = benchmark_method_to_canonical(candidate)

    if canonical_method:
        normalized["canonical_method"] = canonical_method
        normalized["method"] = canonical_method
        normalized["method_label"] = canonical_method_label(canonical_method)

    if raw_benchmark_method is not None:
        normalized["benchmark_method"] = raw_benchmark_method

    if (
        benchmark_origin
        and raw_benchmark_method
        and split
        and not str(normalized.get("artifact_source") or "").strip()
    ):
        normalized["artifact_source"] = f"scores_{raw_benchmark_method}_{split}.csv"

    return normalized


def infer_dataset_from_run(run: str) -> Optional[str]:
    s = run.lower()
    for dataset in DATASET_ORDER:
        if dataset in s:
            return dataset
    return None


def infer_run_kind(run: str) -> str:
    if run.startswith("full_"):
        return "full"
    if run.startswith("smoke_"):
        return "smoke"
    return "legacy"


def dataset_meta(dataset: Optional[str]) -> Optional[Dict[str, str]]:
    if not dataset:
        return None
    return DATASET_INFO.get(dataset, {"label": dataset, "summary": ""})


def split_meta(split: Optional[str]) -> Optional[Dict[str, str]]:
    if not split:
        return None
    return SPLIT_INFO.get(split, {"label": split, "summary": ""})


def view_mode_meta(view_mode: Optional[str]) -> Optional[Dict[str, str]]:
    if not view_mode:
        return None
    return VIEW_MODE_INFO.get(view_mode, {"label": view_mode, "summary": ""})


def benchmark_method_to_api_method(method: str) -> str:
    return benchmark_method_to_canonical(method)


def normalize_capture(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    return _CAPTURE_ALIASES.get(s, s)


def infer_capture_from_path(path: str) -> Optional[str]:
    p = str(path).lower()
    if "contactless" in p or "contact-less" in p or "contact_less" in p:
        return "contactless"
    if "contact-based" in p or "contact_based" in p or "contactbased" in p:
        return "contact_based"
    if "roll" in p:
        return "roll"
    if "plain" in p:
        return "plain"
    return None


def run_sort_key(run: str, validated: bool = False) -> tuple:
    if run in PREFERRED_RUN_ORDER:
        return (0, PREFERRED_RUN_ORDER.index(run))

    dataset = infer_dataset_from_run(run)
    kind = infer_run_kind(run)
    kind_order = {"full": 1, "smoke": 2, "legacy": 3}.get(kind, 9)
    return (
        kind_order,
        DATASET_ORDER.get(dataset or "", 99),
        0 if validated else 1,
        run,
    )


def is_showcase_dataset(dataset: Optional[str]) -> bool:
    return bool(dataset and dataset in SHOWCASE_DATASETS)


def infer_view_mode(run: str, validated: bool = False) -> str:
    if run in CANONICAL_FULL_RUNS and validated:
        return "canonical"
    if run in CANONICAL_SMOKE_RUNS:
        return "smoke"
    return "archive"


def default_split_for_view(available_splits: list[str], view_mode: str) -> Optional[str]:
    if not available_splits:
        return None
    preferred = ["test", "val"] if view_mode != "smoke" else ["val", "test"]
    for key in preferred:
        if key in available_splits:
            return key
    return sorted(available_splits, key=lambda split: SPLIT_ORDER.get(split, 99))[0]


def format_run_label(run: str) -> str:
    if run in CANONICAL_FULL_RUNS:
        return "Canonical full benchmark"
    if run in CANONICAL_SMOKE_RUNS:
        return "Smoke benchmark"
    if run.startswith("full_"):
        return "Archived full benchmark"
    if run.startswith("smoke_"):
        return "Archived smoke benchmark"
    return "Archived benchmark"


def excluded_run_name(run: str) -> bool:
    lower = run.strip().lower()
    blocked_tokens = ("tmp", "tmp2", "scratch", "partial", "broken")
    return any(token in lower for token in blocked_tokens)


def path_is_under(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False
