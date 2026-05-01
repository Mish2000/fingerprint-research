from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import cv2
import numpy as np

from ..visualization.deterministic_preview import (
    ARRAY_SOURCE_SUFFIXES,
    RASTER_SOURCE_SUFFIXES,
    can_render_deterministic_source,
    deterministic_render_strategy,
    read_deterministic_preview_source,
)

ROOT = Path(__file__).resolve().parents[3]

UI_ASSET_GENERATOR_VERSION = "1.0.0"
DEFAULT_MAX_ITEMS_PER_DATASET = 240
THUMBNAIL_SIZE = 160
PREVIEW_SIZE = 512
UI_ASSET_IMAGE_EXT = ".png"

DATASET_LABELS = {
    "l3_sf_v2": "L3-SF v2",
    "nist_sd300b": "NIST SD300B",
    "nist_sd300c": "NIST SD300C",
    "polyu_3d": "PolyU 3D",
    "polyu_cross": "PolyU Cross",
    "unsw_2d3d": "UNSW 2D/3D",
}


@dataclass(frozen=True)
class UiAssetConfig:
    generator_version: str = UI_ASSET_GENERATOR_VERSION
    max_items_per_dataset: Optional[int] = DEFAULT_MAX_ITEMS_PER_DATASET
    thumbnail_size: int = THUMBNAIL_SIZE
    preview_size: int = PREVIEW_SIZE
    thumbnail_background: int = 245
    preview_background: int = 245
    keep_existing_root: bool = False


@dataclass(frozen=True)
class CanonicalUiPolicy:
    name: str
    description: str
    grouping_fields: tuple[str, ...]
    allowed_captures: Optional[frozenset[str]] = None
    allowed_modalities: Optional[frozenset[str]] = None
    require_renderable_sources: bool = False
    fail_closed: bool = False


CANONICAL_UI_POLICIES: Dict[str, CanonicalUiPolicy] = {
    "nist_sd300b": CanonicalUiPolicy(
        name="canonical_plain_roll_only",
        description="Keep only canonical plain and roll captures for browser previews.",
        grouping_fields=("split", "capture"),
        allowed_captures=frozenset({"plain", "roll"}),
    ),
    "nist_sd300c": CanonicalUiPolicy(
        name="canonical_plain_roll_only",
        description="Keep only canonical plain and roll captures for browser previews.",
        grouping_fields=("split", "capture"),
        allowed_captures=frozenset({"plain", "roll"}),
    ),
    "polyu_cross": CanonicalUiPolicy(
        name="canonical_contactless_contactbased_only",
        description="Keep only contactless and contact_based captures for cross-sensor browser previews.",
        grouping_fields=("split", "capture"),
        allowed_captures=frozenset({"contactless", "contact_based"}),
    ),
    "unsw_2d3d": CanonicalUiPolicy(
        name="canonical_optical2d_reconstructed3d_only",
        description="Keep only optical_2d and reconstructed_3d assets; exclude auxiliary derived and intermediate rows.",
        grouping_fields=("split", "modality"),
        allowed_modalities=frozenset({"optical_2d", "reconstructed_3d"}),
        fail_closed=True,
    ),
    "polyu_3d": CanonicalUiPolicy(
        name="canonical_surface_only",
        description="Keep only contactless_3d_surface assets, render deterministic previews from canonical surface artifacts, and fail closed if those renders do not succeed.",
        grouping_fields=("split", "session"),
        allowed_modalities=frozenset({"contactless_3d_surface"}),
        fail_closed=True,
    ),
    "l3_sf_v2": CanonicalUiPolicy(
        name="canonical_synthetic_only",
        description="Keep only the synthetic level-3 modality for browser previews.",
        grouping_fields=("split", "modality"),
        allowed_modalities=frozenset({"synthetic_level3"}),
    ),
}


@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    row_number: int
    source_path_original: str
    source_path_logical: str
    source_path_relative: Optional[str]
    resolved_source_path: Optional[Path]
    split: str
    subject_id: Optional[str]
    finger: Optional[str]
    capture: Optional[str]
    modality: Optional[str]
    impression: Optional[str]
    session: Optional[str]
    sample_id: Optional[str]
    traceability: Dict[str, Any] = field(default_factory=dict)
    manifest_row: Dict[str, str] = field(default_factory=dict)

    @property
    def source_signature(self) -> str:
        return path_signature(self.source_path_logical or self.source_path_original)

    @property
    def group_key(self) -> tuple[str, str, str]:
        return (
            self.split or "unknown",
            self.capture or "unknown",
            self.modality or "unknown",
        )

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.group_key,
            self.subject_id or "",
            self.finger or "",
            self.session or "",
            self.impression or "",
            self.sample_id or "",
            self.source_path_logical,
            self.row_number,
        )


@dataclass(frozen=True)
class BuiltItem:
    asset_id: str
    dataset: str
    split: str
    source_path: str
    thumbnail_path: str
    preview_path: str
    availability_status: str
    subject_id: Optional[str]
    finger: Optional[str]
    capture: Optional[str]
    modality: Optional[str]
    traceability: Dict[str, Any]
    ui_eligible: bool
    selection_reason: str
    selection_policy: str
    render_strategy: str
    original_dimensions: Dict[str, int]
    thumbnail_dimensions: Dict[str, int]
    preview_dimensions: Dict[str, int]


def stable_id(prefix: str, *parts: Any, n: int = 16) -> str:
    text = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:n]}"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _normalize_path(path_str: str) -> str:
    return str(path_str or "").replace("\\", "/").strip()


def _normalize_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _normalize_finger(value: Any) -> Optional[str]:
    text = _normalize_optional(value)
    if text is None:
        return None
    try:
        number = int(float(text))
    except ValueError:
        return text
    return None if number == 0 else str(number)


def path_signature(path_str: str) -> str:
    segments = [segment.lower() for segment in _normalize_path(path_str).split("/") if segment and segment != "."]
    keep = segments[-6:] if len(segments) > 6 else segments
    return "/".join(keep)


def extract_relative_data_path(path_str: str) -> Optional[str]:
    normalized = _normalize_path(path_str)
    lowered = normalized.lower()
    marker = "/data/"
    index = lowered.find(marker)
    if index >= 0:
        return normalized[index + 1 :]
    if lowered.startswith("data/"):
        return normalized
    return None


def _candidate_relative_data_paths(path_str: Optional[str]) -> list[str]:
    normalized = _normalize_path(path_str or "")
    if not normalized:
        return []
    candidates = [normalized]
    lowered = normalized.lower()
    if lowered.startswith("data/") and not lowered.startswith("data/raw/"):
        candidates.append(f"data/raw/{normalized[5:]}")
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _logical_external_source_path(dataset: str, path_str: str) -> str:
    suffix = Path(_normalize_path(path_str)).suffix.lower() or ".bin"
    return f"external_sources/{dataset}/{stable_id('source', path_str)}{suffix}"


def resolve_source_path(repo_root: Path, dataset: str, manifest_path_value: str) -> tuple[str, Optional[str], Optional[Path]]:
    original = _normalize_path(manifest_path_value)
    relative_data_path = extract_relative_data_path(original)
    if relative_data_path:
        for candidate_relative_path in _candidate_relative_data_paths(relative_data_path):
            repo_candidate = (repo_root / candidate_relative_path).resolve()
            if repo_candidate.exists():
                return relative_data_path, candidate_relative_path, repo_candidate

    absolute_candidate = Path(manifest_path_value).expanduser()
    if absolute_candidate.exists():
        if absolute_candidate.is_absolute():
            try:
                relative_to_repo = absolute_candidate.resolve().relative_to(repo_root.resolve())
                relative_str = relative_to_repo.as_posix()
                return relative_str, relative_str, absolute_candidate.resolve()
            except Exception:
                return _logical_external_source_path(dataset, manifest_path_value), relative_data_path, absolute_candidate.resolve()
        return original, relative_data_path, absolute_candidate.resolve()

    logical_path = relative_data_path or _logical_external_source_path(dataset, manifest_path_value)
    return logical_path, relative_data_path, None


def discover_supported_datasets(repo_root: Path = ROOT) -> list[str]:
    manifests_root = repo_root / "data" / "manifests"
    if not manifests_root.exists():
        return []
    datasets = []
    for child in sorted(manifests_root.iterdir()):
        if child.is_dir() and (child / "manifest.csv").exists():
            datasets.append(child.name)
    return datasets


def _iter_manifest_rows(manifest_path: Path) -> Iterator[tuple[int, Dict[str, str]]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            yield row_number, {key: value for key, value in row.items()}


def _load_protocol_note_excerpt(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return " ".join(text.split())[:400]


def _load_manifest_records(repo_root: Path, dataset: str) -> list[SourceRecord]:
    manifest_path = repo_root / "data" / "manifests" / dataset / "manifest.csv"
    records: list[SourceRecord] = []
    for row_number, row in _iter_manifest_rows(manifest_path):
        source_path_original = row.get("path", "")
        source_path_logical, source_path_relative, resolved_path = resolve_source_path(
            repo_root,
            dataset,
            source_path_original,
        )
        subject_id = _normalize_optional(row.get("subject_id"))
        split = _normalize_optional(row.get("split")) or ""
        capture = _normalize_optional(row.get("capture"))
        modality = _normalize_optional(row.get("source_modality"))
        finger = _normalize_finger(row.get("frgp"))
        impression = _normalize_optional(row.get("impression"))
        session = _normalize_optional(row.get("session"))
        sample_id = _normalize_optional(row.get("sample_id"))

        records.append(
            SourceRecord(
                dataset=dataset,
                row_number=row_number,
                source_path_original=source_path_original,
                source_path_logical=source_path_logical,
                source_path_relative=source_path_relative,
                resolved_source_path=resolved_path,
                split=split,
                subject_id=subject_id,
                finger=finger,
                capture=capture,
                modality=modality,
                impression=impression,
                session=session,
                sample_id=sample_id,
                traceability={
                    "manifest_path": f"data/manifests/{dataset}/manifest.csv",
                    "manifest_row_number": row_number,
                    "split": split,
                    "source_path_original": source_path_original,
                    "source_path_relative": source_path_relative,
                },
                manifest_row=row,
            )
        )
    return records


def _record_missing_critical_metadata(record: SourceRecord) -> Optional[str]:
    missing = []
    if not record.source_path_original:
        missing.append("path")
    if not record.split:
        missing.append("split")
    if not record.dataset:
        missing.append("dataset")
    if missing:
        return f"missing critical metadata: {', '.join(missing)}"
    return None


def _selection_policy_label(config: UiAssetConfig) -> str:
    max_items = "all" if config.max_items_per_dataset is None else str(config.max_items_per_dataset)
    return f"deterministic_round_robin_by_split_capture_modality:max_items={max_items}"


def _selection_policy_label_for_dataset(config: UiAssetConfig, dataset: str, policy: Optional[CanonicalUiPolicy]) -> str:
    max_items = "all" if config.max_items_per_dataset is None else str(config.max_items_per_dataset)
    if policy is None:
        return _selection_policy_label(config)
    grouping = ",".join(policy.grouping_fields)
    return f"dataset_canonical_policy[{dataset}:{policy.name}]:group_by={grouping};round_robin:max_items={max_items}"


def _record_group_component(record: SourceRecord, field_name: str) -> str:
    value = getattr(record, field_name, None)
    return str(value or "unknown")


def _record_group_key(record: SourceRecord, policy: Optional[CanonicalUiPolicy]) -> tuple[str, ...]:
    if policy is None:
        return record.group_key
    return tuple(_record_group_component(record, field_name) for field_name in policy.grouping_fields)


def _record_sort_key(record: SourceRecord, policy: Optional[CanonicalUiPolicy]) -> tuple[Any, ...]:
    return (
        _record_group_key(record, policy),
        record.subject_id or "",
        record.finger or "",
        record.capture or "",
        record.modality or "",
        record.session or "",
        record.impression or "",
        record.sample_id or "",
        record.source_path_logical,
        record.row_number,
    )


def _source_is_renderable(record: SourceRecord) -> bool:
    return can_render_deterministic_source(record.resolved_source_path)


def _apply_canonical_ui_policy(
    dataset: str,
    records: Iterable[SourceRecord],
) -> tuple[list[SourceRecord], Optional[CanonicalUiPolicy], Dict[str, int], list[Dict[str, Any]]]:
    policy = CANONICAL_UI_POLICIES.get(dataset)
    stats = {
        "canonical_records_considered": 0,
        "policy_excluded_records": 0,
        "non_renderable_canonical_records": 0,
    }
    exclusions: list[Dict[str, Any]] = []
    if policy is None:
        records_list = list(records)
        stats["canonical_records_considered"] = len(records_list)
        return records_list, None, stats, exclusions

    canonical_records: list[SourceRecord] = []
    for record in records:
        stats["canonical_records_considered"] += 1
        if policy.allowed_captures is not None and (record.capture or "") not in policy.allowed_captures:
            stats["policy_excluded_records"] += 1
            exclusions.append(
                {
                    "row_number": record.row_number,
                    "source_path": record.source_path_logical,
                    "reason": f"excluded by canonical UI policy {policy.name}: capture {record.capture or 'unknown'} is non-canonical",
                }
            )
            continue
        if policy.allowed_modalities is not None and (record.modality or "") not in policy.allowed_modalities:
            stats["policy_excluded_records"] += 1
            exclusions.append(
                {
                    "row_number": record.row_number,
                    "source_path": record.source_path_logical,
                    "reason": f"excluded by canonical UI policy {policy.name}: modality {record.modality or 'unknown'} is non-canonical",
                }
            )
            continue
        if policy.require_renderable_sources and not _source_is_renderable(record):
            stats["non_renderable_canonical_records"] += 1
            exclusions.append(
                {
                    "row_number": record.row_number,
                    "source_path": record.source_path_logical,
                    "reason": f"canonical asset is not directly renderable for UI policy {policy.name}",
                }
            )
            continue
        canonical_records.append(record)
    return canonical_records, policy, stats, exclusions


def _select_records(
    records: Iterable[SourceRecord],
    *,
    max_items: Optional[int],
    policy: Optional[CanonicalUiPolicy] = None,
) -> tuple[list[SourceRecord], Dict[str, int], list[Dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[SourceRecord]] = defaultdict(list)
    stats = {
        "records_considered": 0,
        "duplicates_skipped": 0,
        "missing_source_files": 0,
        "missing_critical_metadata": 0,
    }
    exclusions: list[Dict[str, Any]] = []

    for record in sorted(records, key=lambda item: _record_sort_key(item, policy)):
        stats["records_considered"] += 1
        metadata_error = _record_missing_critical_metadata(record)
        if metadata_error:
            stats["missing_critical_metadata"] += 1
            exclusions.append(
                {
                    "row_number": record.row_number,
                    "source_path": record.source_path_logical,
                    "reason": metadata_error,
                }
            )
            continue
        if record.resolved_source_path is None or not record.resolved_source_path.exists():
            stats["missing_source_files"] += 1
            exclusions.append(
                {
                    "row_number": record.row_number,
                    "source_path": record.source_path_logical,
                    "reason": "source file is missing",
                }
            )
            continue
        grouped[_record_group_key(record, policy)].append(record)

    selected: list[SourceRecord] = []
    seen_source_keys: set[str] = set()
    group_keys = sorted(grouped.keys())
    group_offsets = {key: 0 for key in group_keys}

    while group_keys and (max_items is None or len(selected) < max_items):
        made_progress = False
        for group_key in group_keys:
            rows = grouped[group_key]
            offset = group_offsets[group_key]
            while offset < len(rows):
                candidate = rows[offset]
                offset += 1
                if candidate.source_signature in seen_source_keys:
                    stats["duplicates_skipped"] += 1
                    exclusions.append(
                        {
                            "row_number": candidate.row_number,
                            "source_path": candidate.source_path_logical,
                            "reason": "duplicate source signature skipped",
                        }
                    )
                    continue
                selected.append(candidate)
                seen_source_keys.add(candidate.source_signature)
                group_offsets[group_key] = offset
                made_progress = True
                break
            else:
                group_offsets[group_key] = offset
            if max_items is not None and len(selected) >= max_items:
                break
        if not made_progress:
            break

    return selected, stats, exclusions


def _ensure_safe_output_root(path: Path, repo_root: Path, dataset: str) -> None:
    allowed_parent = (repo_root / "data" / "processed" / dataset).resolve()
    try:
        path.resolve().relative_to(allowed_parent)
    except Exception as exc:
        raise ValueError(f"Refusing to manage ui_assets outside of {allowed_parent}: {path}") from exc


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return

    def _onerror(func: Callable[..., Any], target: str, exc_info: Any) -> None:
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except Exception:
            raise exc_info[1]

    shutil.rmtree(path, onerror=_onerror)


def _prepare_output_root(repo_root: Path, dataset: str, config: UiAssetConfig) -> Path:
    output_root = repo_root / "data" / "processed" / dataset / "ui_assets"
    _ensure_safe_output_root(output_root, repo_root, dataset)
    if output_root.exists() and not config.keep_existing_root:
        _safe_rmtree(output_root)
    (output_root / "thumbnails").mkdir(parents=True, exist_ok=True)
    (output_root / "previews").mkdir(parents=True, exist_ok=True)
    return output_root


def _read_source_image(path: Path) -> np.ndarray:
    image = read_deterministic_preview_source(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape {image.shape} for {path}")


def _render_canvas(image: np.ndarray, target_size: int, background: int) -> tuple[np.ndarray, Dict[str, int]]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")
    scale = min(target_size / width, target_size / height)
    rendered_width = max(1, int(round(width * scale)))
    rendered_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (rendered_width, rendered_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), background, dtype=np.uint8)
    offset_y = (target_size - rendered_height) // 2
    offset_x = (target_size - rendered_width) // 2
    canvas[offset_y : offset_y + rendered_height, offset_x : offset_x + rendered_width] = resized
    return canvas, {"width": rendered_width, "height": rendered_height}


def _write_render(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Failed to write rendered asset: {path}")


def _build_item(
    repo_root: Path,
    record: SourceRecord,
    *,
    config: UiAssetConfig,
    selection_policy: str,
) -> BuiltItem:
    asset_id = stable_id(
        "uiasset",
        record.dataset,
        record.source_path_logical,
        record.split,
        record.subject_id or "",
        record.finger or "",
        record.capture or "",
        record.modality or "",
    )
    thumbnail_relative = Path("data") / "processed" / record.dataset / "ui_assets" / "thumbnails" / f"{asset_id}{UI_ASSET_IMAGE_EXT}"
    preview_relative = Path("data") / "processed" / record.dataset / "ui_assets" / "previews" / f"{asset_id}{UI_ASSET_IMAGE_EXT}"
    thumbnail_output = repo_root / thumbnail_relative
    preview_output = repo_root / preview_relative

    resolved_source = record.resolved_source_path or Path()
    render_strategy = deterministic_render_strategy(resolved_source)
    image = _read_source_image(resolved_source)
    original_dimensions = {"width": int(image.shape[1]), "height": int(image.shape[0])}
    thumbnail_image, thumbnail_dimensions = _render_canvas(image, config.thumbnail_size, config.thumbnail_background)
    preview_image, preview_dimensions = _render_canvas(image, config.preview_size, config.preview_background)
    _write_render(thumbnail_output, thumbnail_image)
    _write_render(preview_output, preview_image)

    selection_reason = (
        "Selected by deterministic round-robin across split/capture/modality groups "
        f"from manifest row {record.row_number}."
    )
    return BuiltItem(
        asset_id=asset_id,
        dataset=record.dataset,
        split=record.split,
        source_path=record.source_path_logical,
        thumbnail_path=thumbnail_relative.as_posix(),
        preview_path=preview_relative.as_posix(),
        availability_status="available",
        subject_id=record.subject_id,
        finger=record.finger,
        capture=record.capture,
        modality=record.modality,
        traceability={
            **record.traceability,
            "thumbnail_path": thumbnail_relative.as_posix(),
            "preview_path": preview_relative.as_posix(),
            "render_strategy": render_strategy,
            "render_source_suffix": resolved_source.suffix.lower(),
        },
        ui_eligible=True,
        selection_reason=selection_reason,
        selection_policy=selection_policy,
        render_strategy=render_strategy,
        original_dimensions=original_dimensions,
        thumbnail_dimensions=thumbnail_dimensions,
        preview_dimensions=preview_dimensions,
    )


def _summarize_items(items: list[BuiltItem], source_records_checked: int, selection_policy: str, validation_status: str) -> Dict[str, Any]:
    items_by_split = Counter(item.split for item in items)
    items_by_capture = Counter(item.capture or "unknown" for item in items)
    items_by_modality = Counter(item.modality or "unknown" for item in items)
    items_by_render_strategy = Counter(item.render_strategy for item in items)
    unique_subjects = len({item.subject_id for item in items if item.subject_id})
    return {
        "source_records_checked": source_records_checked,
        "items_generated": len(items),
        "unique_subjects_selected": unique_subjects,
        "items_by_split": dict(sorted(items_by_split.items())),
        "items_by_capture": dict(sorted(items_by_capture.items())),
        "items_by_modality": dict(sorted(items_by_modality.items())),
        "items_by_render_strategy": dict(sorted(items_by_render_strategy.items())),
        "selection_policy": selection_policy,
        "validation_status": validation_status,
    }


def _validate_dataset_outputs(repo_root: Path, items: list[BuiltItem], report: Dict[str, Any]) -> str:
    missing_thumbnail_paths = []
    missing_preview_paths = []
    for item in items:
        if not (repo_root / item.thumbnail_path).exists():
            missing_thumbnail_paths.append(item.thumbnail_path)
        if not (repo_root / item.preview_path).exists():
            missing_preview_paths.append(item.preview_path)

    report["thumbnail_files_verified"] = len(items) - len(missing_thumbnail_paths)
    report["preview_files_verified"] = len(items) - len(missing_preview_paths)
    report["missing_thumbnail_files"] = missing_thumbnail_paths
    report["missing_preview_files"] = missing_preview_paths

    if not items:
        return "fail"
    if missing_thumbnail_paths or missing_preview_paths:
        return "fail"
    warning_count = (
        report["missing_source_files"]
        + report["unreadable_source_files"]
        + report["missing_critical_metadata"]
        + report["duplicates_skipped"]
    )
    return "pass_with_warnings" if warning_count else "pass"


def build_dataset_ui_assets(
    dataset: str,
    *,
    repo_root: Path = ROOT,
    config: UiAssetConfig = UiAssetConfig(),
) -> Dict[str, Any]:
    manifest_root = repo_root / "data" / "manifests" / dataset
    manifest_path = manifest_root / "manifest.csv"
    split_path = manifest_root / "split.json"
    stats_path = manifest_root / "stats.json"
    protocol_note_path = manifest_root / "protocol_note.md"
    output_root = _prepare_output_root(repo_root, dataset, config)

    records = _load_manifest_records(repo_root, dataset)
    canonical_records, canonical_policy, policy_stats, policy_exclusions = _apply_canonical_ui_policy(dataset, records)
    selected_records, selection_stats, exclusions = _select_records(
        canonical_records,
        max_items=config.max_items_per_dataset,
        policy=canonical_policy,
    )
    exclusions = [*policy_exclusions, *exclusions]

    items: list[BuiltItem] = []
    unreadable_source_files = 0
    selection_policy = _selection_policy_label_for_dataset(config, dataset, canonical_policy)
    for record in selected_records:
        try:
            items.append(
                _build_item(
                    repo_root,
                    record,
                    config=config,
                    selection_policy=selection_policy,
                )
            )
        except Exception as exc:
            unreadable_source_files += 1
            exclusions.append(
                {
                    "row_number": record.row_number,
                    "source_path": record.source_path_logical,
                    "reason": f"failed to render preview assets: {exc}",
                }
            )

    report: Dict[str, Any] = {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS.get(dataset, dataset.replace("_", " ").title()),
        "generated_at": _now_utc(),
        "generator_version": config.generator_version,
        "selection_policy": selection_policy,
        "deterministic_render_source_suffixes": sorted((*RASTER_SOURCE_SUFFIXES, *ARRAY_SOURCE_SUFFIXES)),
        "source_records_checked": len(records),
        "canonical_records_considered": policy_stats["canonical_records_considered"],
        "selected_records": len(selected_records),
        "canonical_render_attempts": len(selected_records),
        "canonical_render_failures": unreadable_source_files,
        "generated_items": len(items),
        "excluded_records": len(exclusions),
        "policy_excluded_records": policy_stats["policy_excluded_records"],
        "non_renderable_canonical_records": policy_stats["non_renderable_canonical_records"],
        "rendered_items_by_strategy": dict(sorted(Counter(item.render_strategy for item in items).items())),
        "missing_source_files": selection_stats["missing_source_files"],
        "unreadable_source_files": unreadable_source_files,
        "missing_critical_metadata": selection_stats["missing_critical_metadata"],
        "duplicates_skipped": selection_stats["duplicates_skipped"],
        "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "split_path": split_path.relative_to(repo_root).as_posix() if split_path.exists() else None,
        "stats_path": stats_path.relative_to(repo_root).as_posix() if stats_path.exists() else None,
        "protocol_note_path": protocol_note_path.relative_to(repo_root).as_posix() if protocol_note_path.exists() else None,
        "canonical_ui_policy": (
            {
                "name": canonical_policy.name,
                "description": canonical_policy.description,
                "grouping_fields": list(canonical_policy.grouping_fields),
                "allowed_captures": sorted(canonical_policy.allowed_captures) if canonical_policy.allowed_captures else None,
                "allowed_modalities": sorted(canonical_policy.allowed_modalities) if canonical_policy.allowed_modalities else None,
                "require_renderable_sources": canonical_policy.require_renderable_sources,
                "fail_closed": canonical_policy.fail_closed,
            }
            if canonical_policy is not None
            else None
        ),
        "exclusions": exclusions[:200],
    }
    validation_status = _validate_dataset_outputs(repo_root, items, report)
    report["validation_status"] = validation_status
    if canonical_policy is not None and canonical_policy.fail_closed and not items:
        report["browser_preview_enabled"] = False
        if selected_records:
            report["browser_preview_exclusion_reason"] = (
                f"No canonical preview assets rendered successfully for {dataset} after "
                f"{len(selected_records)} deterministic render attempt(s)."
            )
        else:
            report["browser_preview_exclusion_reason"] = (
                f"No canonical assets satisfied {canonical_policy.name} for {dataset}."
            )
    else:
        report["browser_preview_enabled"] = validation_status != "fail"
        report["browser_preview_exclusion_reason"] = None if report["browser_preview_enabled"] else "dataset UI asset validation failed"

    index_payload = {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS.get(dataset, dataset.replace("_", " ").title()),
        "generated_at": report["generated_at"],
        "generator_version": config.generator_version,
        "selection_policy": selection_policy,
        "deterministic_render_source_suffixes": report["deterministic_render_source_suffixes"],
        "items": [asdict(item) for item in items],
        "summary": _summarize_items(items, len(records), selection_policy, validation_status),
        "validation_status": validation_status,
        "browser_preview_enabled": report["browser_preview_enabled"],
        "browser_preview_exclusion_reason": report["browser_preview_exclusion_reason"],
        "canonical_ui_policy": report["canonical_ui_policy"],
        "stats_path": report["stats_path"],
        "split_path": report["split_path"],
        "protocol_note_path": report["protocol_note_path"],
        "protocol_note_excerpt": _load_protocol_note_excerpt(protocol_note_path),
    }

    _json_dump(output_root / "index.json", index_payload)
    _json_dump(output_root / "validation_report.json", report)

    return {
        "dataset": dataset,
        "index_path": (output_root / "index.json").relative_to(repo_root).as_posix(),
        "validation_report_path": (output_root / "validation_report.json").relative_to(repo_root).as_posix(),
        "item_count": len(items),
        "validation_status": validation_status,
        "selection_policy": selection_policy,
        "browser_preview_enabled": report["browser_preview_enabled"],
        "browser_preview_exclusion_reason": report["browser_preview_exclusion_reason"],
        "summary": index_payload["summary"],
    }


def build_ui_assets(
    datasets: Optional[Iterable[str]] = None,
    *,
    repo_root: Path = ROOT,
    config: UiAssetConfig = UiAssetConfig(),
) -> Dict[str, Any]:
    selected_datasets = list(datasets or discover_supported_datasets(repo_root))
    results = []
    for dataset in selected_datasets:
        results.append(build_dataset_ui_assets(dataset, repo_root=repo_root, config=config))

    registry = {
        "generated_at": _now_utc(),
        "generator_version": config.generator_version,
        "datasets": results,
    }
    _json_dump(repo_root / "data" / "processed" / "ui_assets_registry.json", registry)
    return registry
