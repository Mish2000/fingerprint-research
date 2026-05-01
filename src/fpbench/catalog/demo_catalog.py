from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field

from apps.api.benchmark_meta import (
    benchmark_method_to_canonical,
    benchmark_row_is_current,
    benchmark_row_semantics_epoch,
    normalize_benchmark_context,
)
from apps.api.schemas import MatchMethod

from ..visualization.deterministic_preview import (
    read_deterministic_preview_source,
)


CATALOG_VERSION = "1.0.0"
GENERATION_POLICY_VERSION = "1.0.0"
VERIFY_METHOD_ENUM = [method.value for method in MatchMethod]
VERIFY_METHOD_SET = frozenset(VERIFY_METHOD_ENUM)
AVAILABILITY_ENUM = ["available", "missing"]
ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_ROOT = ROOT / "data" / "manifests"
BENCH_ROOT = ROOT / "artifacts" / "reports" / "benchmark"
SAMPLES_ROOT = ROOT / "data" / "samples"
BEST_METHODS_JSON = BENCH_ROOT / "april_comparison" / "best_methods.json"
ASSETS_ROOT = SAMPLES_ROOT / "assets"
MATERIALIZED_ASSET_KIND = "binary_image"
MATERIALIZED_ASSET_EXT = ".png"
MATERIALIZED_ASSET_THUMBNAIL_SIZE = 192
BENCHMARK_SHOWCASE_RUNS: Dict[str, Tuple[str, ...]] = {
    "nist_sd300b": ("full_nist_sd300b_h6", "full_nist_sd300b"),
    "nist_sd300c": ("full_nist_sd300c_h6", "full_nist_sd300c"),
    "polyu_cross": ("full_polyu_cross_h5", "full_polyu_cross"),
}
BENCHMARK_EXCLUDED_RUN_TOKENS = ("tmp", "tmp2", "scratch", "partial", "broken")


DATASET_META: Dict[str, Dict[str, Any]] = {
    "nist_sd300b": {
        "label": "NIST SD300B",
        "protocol": "verification",
        "default_relation": "same_modality",
        "safe_default": False,
        "fallback_method": "sift",
        "notes": "Plain-to-roll public benchmark dataset at 1000 ppi.",
    },
    "nist_sd300c": {
        "label": "NIST SD300C",
        "protocol": "verification",
        "default_relation": "same_modality",
        "safe_default": False,
        "fallback_method": "sift",
        "notes": "Plain-to-roll public benchmark dataset at 2000 ppi.",
    },
    "polyu_cross": {
        "label": "PolyU Cross",
        "protocol": "cross_modality",
        "default_relation": "cross_modality",
        "safe_default": False,
        "fallback_method": "dl",
        "notes": "Contactless-to-contact-based cross-modality dataset.",
    },
    "unsw_2d3d": {
        "label": "UNSW 2D/3D",
        "protocol": "cross_modality",
        "default_relation": "cross_modality",
        "safe_default": False,
        "fallback_method": "dl",
        "notes": "2D optical to reconstructed 3D cross-modality dataset.",
    },
    "polyu_3d": {
        "label": "PolyU 3D",
        "protocol": "surface_only",
        "default_relation": "same_modality",
        "safe_default": False,
        "fallback_method": "dedicated",
        "notes": "Cross-session 3D surface bundle with first-session gallery, second-session probe, and frgp placeholder 0.",
    },
    "l3_sf_v2": {
        "label": "L3-SF v2",
        "protocol": "synthetic_level3",
        "default_relation": "same_modality",
        "safe_default": True,
        "fallback_method": "dedicated",
        "notes": "Synthetic level-3 fingerprint dataset.",
    },
}


VERIFY_CASE_PLANS: List[Dict[str, Any]] = [
    {
        "case_id": "easy_genuine_nist_1000",
        "dataset": "nist_sd300b",
        "split": "val",
        "case_type": "easy_genuine",
        "label": 1,
        "difficulty": "easy",
        "selector": "benchmark_top",
        "demo_safe": True,
    },
    {
        "case_id": "hard_genuine_nist_1000",
        "dataset": "nist_sd300b",
        "split": "val",
        "case_type": "hard_genuine",
        "label": 1,
        "difficulty": "hard",
        "selector": "benchmark_bottom",
        "demo_safe": True,
    },
    {
        "case_id": "hard_impostor_nist_1000",
        "dataset": "nist_sd300b",
        "split": "val",
        "case_type": "hard_impostor",
        "label": 0,
        "difficulty": "hard",
        "selector": "benchmark_top",
        "demo_safe": True,
    },
    {
        "case_id": "easy_genuine_nist_2000",
        "dataset": "nist_sd300c",
        "split": "test",
        "case_type": "easy_genuine",
        "label": 1,
        "difficulty": "easy",
        "selector": "benchmark_top",
        "demo_safe": True,
    },
    {
        "case_id": "hard_impostor_nist_2000",
        "dataset": "nist_sd300c",
        "split": "test",
        "case_type": "hard_impostor",
        "label": 0,
        "difficulty": "hard",
        "selector": "benchmark_top",
        "demo_safe": True,
    },
    {
        "case_id": "cross_modality_genuine_crossfingerprint",
        "dataset": "polyu_cross",
        "split": "val",
        "case_type": "cross_modality_genuine",
        "label": 1,
        "difficulty": "medium",
        "selector": "benchmark_top",
        "demo_safe": True,
    },
    {
        "case_id": "cross_modality_impostor_crossfingerprint",
        "dataset": "polyu_cross",
        "split": "val",
        "case_type": "cross_modality_impostor",
        "label": 0,
        "difficulty": "hard",
        "selector": "benchmark_top",
        "demo_safe": True,
    },
]

VERIFY_DEMO_DATASETS = frozenset(str(plan["dataset"]) for plan in VERIFY_CASE_PLANS)
IDENTITY_DEMO_SAFE_DATASETS = frozenset({"l3_sf_v2"})
IDENTITY_DATASET_ORDER = ["nist_sd300b", "nist_sd300c", "polyu_cross", "unsw_2d3d", "polyu_3d", "l3_sf_v2"]


class AvailabilityDetail(BaseModel):
    status: Literal["available", "missing"]
    local_exists: bool
    traceable_to_manifest: bool
    relative_path: Optional[str] = None
    resolved_local_path: Optional[str] = None
    source_relative_path: Optional[str] = None
    source_resolved_local_path: Optional[str] = None
    source_local_exists: bool = False


class CatalogAsset(BaseModel):
    asset_id: str
    dataset: str
    path: str
    relative_path: Optional[str] = None
    source_path: Optional[str] = None
    source_relative_path: Optional[str] = None
    signature: str
    subject_id: int
    capture: str
    finger: str
    split: str
    session: Optional[int] = None
    impression: Optional[str] = None
    source_modality: Optional[str] = None
    availability_status: Literal["available", "missing"]
    availability_detail: AvailabilityDetail
    materialized_asset_kind: Optional[str] = None
    thumbnail_path: Optional[str] = None
    content_type: Optional[str] = None
    dimensions: Optional[Dict[str, int]] = None
    thumbnail_dimensions: Optional[Dict[str, int]] = None
    traceability: Dict[str, Any] = Field(default_factory=dict)
    thumbnail_hint: Optional[str] = None
    quality_hint: Optional[str] = None
    recommended_usage: Optional[str] = None


class VerifyCase(BaseModel):
    case_id: str
    title: str
    description: str
    dataset: str
    split: str
    case_type: str
    difficulty: Literal["easy", "medium", "hard", "challenging"]
    ground_truth: Literal["match", "non_match"]
    image_a: CatalogAsset
    image_b: CatalogAsset
    subject_a: int
    subject_b: int
    finger_a: str
    finger_b: str
    capture_a: str
    capture_b: str
    modality_relation: Literal["same_modality", "cross_modality", "unknown"]
    source_pair_file: str
    source_pair_row_id: str
    recommended_method: MatchMethod
    tags: List[str] = Field(default_factory=list)
    is_demo_safe: bool
    availability_status: Literal["available", "traceable_only", "missing"]
    availability_detail: Dict[str, Any] = Field(default_factory=dict)
    selection_reason: str
    selection_policy: str
    benchmark_context: Optional[Dict[str, Any]] = None
    selection_diagnostics: Optional[Dict[str, Any]] = None


class IdentityRecord(BaseModel):
    identity_id: str
    dataset: str
    display_name: str
    subject_id: int
    gallery_role: str
    enrollment_candidates: List[str]
    probe_candidates: List[str]
    recommended_enrollment_asset_id: Optional[str] = None
    recommended_probe_asset_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    is_demo_safe: bool
    exemplars: List[CatalogAsset] = Field(default_factory=list)


class IdentificationScenario(BaseModel):
    scenario_id: str
    scenario_type: Literal["positive_identification", "difficult_identification", "no_match"]
    dataset: str
    title: str
    description: str
    enrollment_identity_ids: List[str] = Field(default_factory=list)
    probe_asset: CatalogAsset
    expected_identity_id: Optional[str] = None
    difficulty: Literal["easy", "medium", "hard", "challenging"]
    recommended_method: MatchMethod
    tags: List[str] = Field(default_factory=list)
    is_demo_safe: bool


class IdentifyGallery(BaseModel):
    identities: List[IdentityRecord] = Field(default_factory=list)
    demo_scenarios: List[IdentificationScenario] = Field(default_factory=list)


class BrowserSeedEntry(BaseModel):
    dataset: str
    items: List[CatalogAsset] = Field(default_factory=list)
    selection_policy: str
    coverage_summary: Dict[str, Any] = Field(default_factory=dict)


class SourceDataset(BaseModel):
    dataset: str
    dataset_label: str
    manifest_path: str
    pair_files: Dict[str, str]
    stats_path: str
    split_path: str
    protocol_note_path: Optional[str] = None
    benchmark_runs: List[Dict[str, Any]] = Field(default_factory=list)
    manifest_rows: int
    unique_subjects: int
    included_in_catalog: bool
    exclusion_reason: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    verify_selection_diagnostics: List[Dict[str, Any]] = Field(default_factory=list)


class Metadata(BaseModel):
    catalog_version: str
    generation_policy_version: str
    total_verify_cases: int
    total_identity_records: int
    total_browser_seed_items: int
    included_datasets: List[str]
    excluded_datasets: List[Dict[str, str]] = Field(default_factory=list)
    validation_status: str
    validation_errors_count: int
    validation_warnings_count: int
    materialized_asset_root: Optional[str] = None
    materialized_asset_count: int = 0
    limitations: List[str] = Field(default_factory=list)
    catalog_build_health: Dict[str, Any] = Field(default_factory=dict)


class CatalogModel(BaseModel):
    catalog_version: str
    generated_at: str
    source_datasets: List[SourceDataset]
    verify_cases: List[VerifyCase]
    identify_gallery: IdentifyGallery
    dataset_browser_seed: List[BrowserSeedEntry]
    metadata: Metadata


@dataclass
class DatasetBundle:
    dataset: str
    manifest: pd.DataFrame
    pairs_by_split: Dict[str, pd.DataFrame]
    stats: Dict[str, Any]
    split_meta: Dict[str, Any]
    protocol_note_path: Optional[str]
    manifest_lookup: Dict[str, Dict[str, Any]]
    manifest_lookup_by_signature: Dict[str, Dict[str, Any]]
    benchmark_best: Dict[str, Dict[str, Any]]


@dataclass
class SelectionResult:
    row: pd.Series
    score: Optional[float]
    benchmark_context: Optional[Dict[str, Any]]
    selection_diagnostics: Dict[str, Any]
    selection_policy: str
    selection_reason: str



def stable_id(prefix: str, *parts: Any, n: int = 12) -> str:
    text = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:n]}"



def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")



def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))



def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}



def _parse_json_text(raw: Any) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}



def _normalize_dataset_key(raw: Any) -> Optional[str]:
    if isinstance(raw, dict):
        return _normalize_dataset_key(raw.get("name"))
    text = str(raw or "").strip().lower()
    if not text:
        return None
    return text if text in DATASET_META else None



def _normalize_segments(path_str: str) -> List[str]:
    if not path_str:
        return []
    p = str(path_str).replace("\\", "/")
    return [seg for seg in p.split("/") if seg and seg not in {"."}]



def _extract_relative_data_path(path_str: str) -> Optional[str]:
    if not path_str:
        return None
    path_norm = str(path_str).replace("\\", "/")
    lowered = path_norm.lower()
    marker = "/data/"
    idx = lowered.find(marker)
    if idx >= 0:
        suffix = path_norm[idx + 1 :]
        return suffix.replace("\\", "/")
    if lowered.startswith("data/"):
        return path_norm
    return None



def _candidate_local_path(relative_data_path: Optional[str]) -> Optional[Path]:
    if not relative_data_path:
        return None
    normalized = str(relative_data_path).replace("\\", "/").strip()
    candidates = [normalized]
    lowered = normalized.lower()
    if lowered.startswith("data/") and not lowered.startswith("data/raw/"):
        candidates.append(f"data/raw/{normalized[5:]}")

    resolved_candidates = [(ROOT / candidate).resolve() for candidate in candidates]
    for candidate in resolved_candidates:
        if candidate.exists():
            return candidate
    return resolved_candidates[0]



def path_signature(path_str: str, n: int = 4) -> str:
    segs = [seg.lower() for seg in _normalize_segments(path_str)]
    if not segs:
        return ""
    depth = max(int(n), 1)
    keep = segs[-depth:] if len(segs) >= depth else segs
    return "/".join(keep)


def _score_merge_with_signature_depth(
    pairs: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    signature_depth: int,
) -> pd.DataFrame:
    scored = scores.copy()
    scored["sig_a"] = scored["path_a"].map(lambda value: path_signature(value, n=signature_depth))
    scored["sig_b"] = scored["path_b"].map(lambda value: path_signature(value, n=signature_depth))
    scored["pair_key"] = scored["sig_a"] + "||" + scored["sig_b"]

    pair_frame = pairs.copy()
    pair_frame["sig_a"] = pair_frame["path_a"].map(lambda value: path_signature(value, n=signature_depth))
    pair_frame["sig_b"] = pair_frame["path_b"].map(lambda value: path_signature(value, n=signature_depth))
    pair_frame["pair_key"] = pair_frame["sig_a"] + "||" + pair_frame["sig_b"]
    return pair_frame.merge(scored[["pair_key", "score"]], on="pair_key", how="inner")



def parse_numeric_suffix(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None



def _maybe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(numeric) else numeric



def infer_modality_relation(row_a: Dict[str, Any], row_b: Dict[str, Any], dataset: str) -> str:
    dataset_meta = DATASET_META.get(dataset, {})
    dataset_default = str(dataset_meta.get("default_relation", "unknown"))
    protocol = str(dataset_meta.get("protocol", ""))

    mod_a = str(row_a.get("source_modality", "") or "").strip()
    mod_b = str(row_b.get("source_modality", "") or "").strip()

    if mod_a and mod_b:
        if mod_a == mod_b:
            return "same_modality"
        if protocol == "cross_modality":
            return "cross_modality"
        return dataset_default

    return dataset_default



def availability_detail_for_path(path_str: str, traceable_to_manifest: bool = True) -> AvailabilityDetail:
    rel = _extract_relative_data_path(path_str)
    local = _candidate_local_path(rel)
    local_exists = bool(local and local.exists())
    status = "available" if local_exists else "missing"
    return AvailabilityDetail(
        status=status,
        local_exists=local_exists,
        traceable_to_manifest=traceable_to_manifest,
        relative_path=rel,
        resolved_local_path=str(local) if local else None,
        source_relative_path=rel,
        source_resolved_local_path=str(local) if local else None,
        source_local_exists=local_exists,
    )



def build_asset(dataset: str, manifest_row: Dict[str, Any], *, recommended_usage: Optional[str] = None, quality_hint: Optional[str] = None) -> CatalogAsset:
    path = str(manifest_row.get("path"))
    detail = availability_detail_for_path(path, traceable_to_manifest=True)
    finger_value = manifest_row.get("frgp", "unknown")
    finger = "unknown" if pd.isna(finger_value) or int(finger_value) == 0 else str(int(finger_value))
    relative = detail.relative_path
    traceability = {
        "source_dataset": dataset,
        "source_manifest_path": f"data/manifests/{dataset}/manifest.csv",
        "source_path_signature": path_signature(path),
        "source_subject_id": int(manifest_row.get("subject_id", -1)),
        "source_split": str(manifest_row.get("split", "unknown")),
    }
    return CatalogAsset(
        asset_id=stable_id("asset", dataset, path_signature(path), manifest_row.get("subject_id"), finger, manifest_row.get("capture")),
        dataset=dataset,
        path=path,
        relative_path=relative,
        source_path=path,
        source_relative_path=relative,
        signature=path_signature(path),
        subject_id=int(manifest_row.get("subject_id")),
        capture=str(manifest_row.get("capture", "unknown")),
        finger=finger,
        split=str(manifest_row.get("split", "unknown")),
        session=int(manifest_row.get("session")) if str(manifest_row.get("session", "")).isdigit() else None,
        impression=str(manifest_row.get("impression")) if manifest_row.get("impression") is not None else None,
        source_modality=str(manifest_row.get("source_modality")) if manifest_row.get("source_modality") is not None else None,
        availability_status=detail.status,
        availability_detail=detail,
        materialized_asset_kind=None,
        traceability=traceability,
        thumbnail_hint=None,
        quality_hint=quality_hint,
        recommended_usage=recommended_usage,
    )



def _canonical_verify_method_or_none(method: Any) -> Optional[str]:
    canonical = benchmark_method_to_canonical(method)
    return canonical if canonical in VERIFY_METHOD_SET else None



def recommended_method_for_dataset(bundle: DatasetBundle, split: str) -> Tuple[str, Dict[str, Any]]:
    best = resolve_benchmark_best_for_dataset_split(bundle, split)
    best_auc = best.get("best_auc")
    if best_auc:
        raw_method = str(best_auc.get("method") or "").strip()
        if raw_method:
            canonical_method = _canonical_verify_method_or_none(raw_method)
            if canonical_method:
                context = {
                    "source": "benchmark_best_auc",
                    "run": best_auc.get("run"),
                    "method": canonical_method,
                    "benchmark_method": raw_method,
                    "canonical_method": canonical_method,
                    "benchmark_best_source": best_auc.get("benchmark_best_source") or best.get("benchmark_best_source"),
                }
                if best_auc.get("scores_csv"):
                    context["scores_csv"] = best_auc.get("scores_csv")
                if best_auc.get("summary_csv"):
                    context["summary_csv"] = best_auc.get("summary_csv")
                return canonical_method, context
    fallback = str(DATASET_META[bundle.dataset]["fallback_method"])
    return fallback, {
        "source": "dataset_fallback",
        "method": fallback,
        "canonical_method": fallback,
        "benchmark_method": None,
    }



def _sort_pairs_for_selection(df: pd.DataFrame, ascending: bool, tie_cols: Optional[List[str]] = None) -> pd.DataFrame:
    cols = ["score"] if "score" in df.columns else []
    if tie_cols:
        cols.extend(tie_cols)
    if not cols:
        return df.copy()
    asc = [ascending] + [True] * (len(cols) - 1) if "score" in df.columns else [True] * len(cols)
    return df.sort_values(cols, ascending=asc, kind="mergesort")



def _heuristic_challenge_score(row: pd.Series, bundle: DatasetBundle) -> float:
    row_a = bundle.manifest_lookup.get(str(row["path_a"])) or bundle.manifest_lookup_by_signature.get(path_signature(str(row["path_a"]))) or {}
    row_b = bundle.manifest_lookup.get(str(row["path_b"])) or bundle.manifest_lookup_by_signature.get(path_signature(str(row["path_b"]))) or {}
    session_gap = abs(int(row_a.get("session", 0) or 0) - int(row_b.get("session", 0) or 0))
    impression_gap = abs((parse_numeric_suffix(row_a.get("impression")) or 0) - (parse_numeric_suffix(row_b.get("impression")) or 0))
    modality_gap = 1 if infer_modality_relation(row_a, row_b, bundle.dataset) == "cross_modality" else 0
    variant_rank = int(row_a.get("variant_rank", 0) or 0) + int(row_b.get("variant_rank", 0) or 0)
    unknown_finger = 1 if int(row.get("frgp", 0) or 0) == 0 else 0
    subject_gap = abs(int(row.get("subject_a", 0)) - int(row.get("subject_b", 0)))
    return float(modality_gap * 100 + session_gap * 10 + impression_gap * 2 + variant_rank + unknown_finger - (subject_gap / 1000000.0))



def _heuristic_negative_difficulty(row: pd.Series, bundle: DatasetBundle) -> float:
    row_a = bundle.manifest_lookup.get(str(row["path_a"])) or bundle.manifest_lookup_by_signature.get(path_signature(str(row["path_a"]))) or {}
    row_b = bundle.manifest_lookup.get(str(row["path_b"])) or bundle.manifest_lookup_by_signature.get(path_signature(str(row["path_b"]))) or {}
    subject_gap = abs(int(row.get("subject_a", 0)) - int(row.get("subject_b", 0)))
    session_gap = abs(int(row_a.get("session", 0) or 0) - int(row_b.get("session", 0) or 0))
    impression_gap = abs((parse_numeric_suffix(row_a.get("impression")) or 0) - (parse_numeric_suffix(row_b.get("impression")) or 0))
    modality_gap = 1 if infer_modality_relation(row_a, row_b, bundle.dataset) == "cross_modality" else 0
    return float((1000 - min(subject_gap, 1000)) + modality_gap * 100 - session_gap * 5 - impression_gap)


def _reportable_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT).as_posix())
    except ValueError:
        return str(path)


def _benchmark_discovery_outcome(method_context: Dict[str, Any]) -> str:
    source = str(method_context.get("source") or "").strip()
    run_name = str(method_context.get("run") or "").strip()
    raw_method = str(method_context.get("benchmark_method") or "").strip()
    canonical_method = str(method_context.get("canonical_method") or method_context.get("method") or "").strip()
    if source == "benchmark_best_auc":
        if run_name and raw_method and canonical_method:
            return "benchmark_best_resolved"
        return "benchmark_best_incomplete"
    if source == "dataset_fallback":
        return "dataset_fallback_no_benchmark_evidence"
    if source:
        return source
    return "unknown"


def _build_selection_diagnostics(
    *,
    plan: Dict[str, Any],
    method_context: Dict[str, Any],
) -> Dict[str, Any]:
    canonical_method = (
        str(method_context.get("canonical_method") or method_context.get("method") or "").strip() or None
    )
    raw_benchmark_method = str(method_context.get("benchmark_method") or "").strip() or None
    chosen_run = str(method_context.get("run") or "").strip() or None
    return {
        "selector": str(plan["selector"]),
        "selection_driver": "pending",
        "benchmark_discovery_outcome": _benchmark_discovery_outcome(method_context),
        "benchmark_best_source": method_context.get("benchmark_best_source"),
        "benchmark_selection_status": "pending",
        "fallback_reason": None,
        "fallback_category": None,
        "chosen_run": chosen_run,
        "chosen_raw_benchmark_method": raw_benchmark_method,
        "chosen_canonical_method": canonical_method,
        "score_artifact_path_used": None,
        "benchmark_score": None,
        "benchmark_pair_match_signature_depth": None,
        "benchmark_backed_selection": False,
        "heuristic_fallback_used": False,
    }


def _fallback_selection_from_heuristics(
    bundle: DatasetBundle,
    pairs: pd.DataFrame,
    *,
    label: int,
    selector: str,
    run_name: Optional[str],
    bench_method: Optional[str],
    selection_diagnostics: Dict[str, Any],
    fallback_category: str,
    fallback_reason: str,
) -> Optional[SelectionResult]:
    scored = pairs.copy()
    if label == 1:
        scored["score"] = scored.apply(lambda row: _heuristic_challenge_score(row, bundle), axis=1)
        choose_harder_case = selector.endswith("bottom")
        fallback_policy = "heuristic challenge fallback"
        fallback_summary = "heuristic challenge ranking"
    else:
        scored["score"] = scored.apply(lambda row: _heuristic_negative_difficulty(row, bundle), axis=1)
        choose_harder_case = selector.endswith("top")
        fallback_policy = "heuristic impostor fallback"
        fallback_summary = "heuristic impostor difficulty ranking"

    if scored.empty:
        return None

    scored = _sort_pairs_for_selection(
        scored,
        ascending=not choose_harder_case,
        tie_cols=["path_a", "path_b"],
    )
    row = scored.iloc[0]
    diagnostics = dict(selection_diagnostics)
    diagnostics.update(
        {
            "selection_driver": "heuristic_fallback",
            "benchmark_selection_status": fallback_category,
            "fallback_reason": fallback_reason,
            "fallback_category": fallback_category,
            "heuristic_fallback_used": True,
        }
    )
    benchmark_context = {
        "selection_driver": "heuristic_fallback",
        "benchmark_run": run_name,
        "benchmark_method": bench_method,
        "selection_mode": selector,
        "benchmark_fallback_reason": fallback_reason,
        "heuristic_basis": fallback_summary,
    }
    return SelectionResult(
        row=row,
        score=float(row["score"]),
        benchmark_context=benchmark_context,
        selection_diagnostics=diagnostics,
        selection_policy=f"{fallback_policy} ({selector} degraded gracefully)",
        selection_reason=(
            f"Benchmark-driven selector {selector} could not use benchmark evidence directly; "
            f"falling back to {fallback_summary}. Reason: {fallback_reason}"
        ),
    )



def select_case_row(bundle: DatasetBundle, plan: Dict[str, Any]) -> Optional[SelectionResult]:
    split = str(plan["split"])
    label = int(plan["label"])
    selector = str(plan["selector"])
    pairs = bundle.pairs_by_split.get(split)
    if pairs is None or pairs.empty:
        return None
    pairs = pairs[pairs["label"].astype(int) == label].copy()
    if pairs.empty:
        return None

    if selector.startswith("benchmark"):
        best = resolve_benchmark_best_for_dataset_split(bundle, split)
        _, method_context = recommended_method_for_dataset(bundle, split)
        run_name = str(method_context.get("run") or "").strip() or None
        bench_method = str(method_context.get("benchmark_method") or "").strip() or None
        selection_diagnostics = _build_selection_diagnostics(plan=plan, method_context=method_context)
        if run_name and bench_method:
            score_filename = str(method_context.get("scores_csv") or best.get("best_auc", {}).get("scores_csv") or f"scores_{bench_method}_{split}.csv")
            scores_path = BENCH_ROOT / str(run_name) / score_filename
            selection_diagnostics["score_artifact_path_used"] = _reportable_path(scores_path)
            if scores_path.exists():
                try:
                    scores = pd.read_csv(scores_path)
                except Exception as exc:
                    fallback = _fallback_selection_from_heuristics(
                        bundle,
                        pairs,
                        label=label,
                        selector=selector,
                        run_name=str(run_name),
                        bench_method=str(bench_method),
                        selection_diagnostics=selection_diagnostics,
                        fallback_category="score_file_unparsable",
                        fallback_reason=f"Score file {scores_path.name} could not be parsed: {exc}",
                    )
                    if fallback is not None:
                        return fallback
                required_score_columns = {"label", "path_a", "path_b", "score"}
                if not required_score_columns.issubset(set(scores.columns)):
                    fallback = _fallback_selection_from_heuristics(
                        bundle,
                        pairs,
                        label=label,
                        selector=selector,
                        run_name=str(run_name),
                        bench_method=str(bench_method),
                        selection_diagnostics=selection_diagnostics,
                        fallback_category="score_file_missing_columns",
                        fallback_reason=(
                            f"Score file {scores_path.name} is unusable for pair selection; "
                            f"missing columns: {sorted(required_score_columns - set(scores.columns))}."
                        ),
                    )
                    if fallback is not None:
                        return fallback
                scores = scores[scores["label"].astype(int) == label].copy()
                merged = pd.DataFrame()
                matched_signature_depth = None
                for signature_depth in (4, 3):
                    merged = _score_merge_with_signature_depth(
                        pairs,
                        scores,
                        signature_depth=signature_depth,
                    )
                    if not merged.empty:
                        matched_signature_depth = signature_depth
                        break

                if not merged.empty:
                    ascending = selector.endswith("bottom")
                    merged = _sort_pairs_for_selection(merged, ascending=ascending, tie_cols=["path_a", "path_b"])
                    row = merged.iloc[0]
                    diagnostics = dict(selection_diagnostics)
                    diagnostics.update(
                        {
                            "selection_driver": "benchmark_driven",
                            "benchmark_selection_status": "benchmark_score_used",
                            "benchmark_score": float(row["score"]),
                            "benchmark_pair_match_signature_depth": matched_signature_depth,
                            "benchmark_backed_selection": True,
                        }
                    )
                    run_info = {
                        "selection_driver": "benchmark_driven",
                        "benchmark_run": run_name,
                        "benchmark_method": bench_method,
                        "benchmark_score": float(row["score"]),
                        "artifact_source": scores_path.name,
                        "score_artifact_path_used": _reportable_path(scores_path),
                        "selection_mode": selector,
                        "benchmark_pair_match_signature_depth": matched_signature_depth,
                    }
                    return SelectionResult(
                        row=row,
                        score=float(row["score"]),
                        benchmark_context=run_info,
                        selection_diagnostics=diagnostics,
                        selection_policy=(
                            f"{selector} via {run_name}/{bench_method}"
                            f" (pair_match_signature_depth={matched_signature_depth})"
                        ),
                        selection_reason=(
                            f"Selected by benchmark score ranking ({selector}) using {bench_method} on {split}."
                        ),
                    )
                fallback = _fallback_selection_from_heuristics(
                    bundle,
                    pairs,
                    label=label,
                    selector=selector,
                    run_name=str(run_name),
                    bench_method=str(bench_method),
                    selection_diagnostics=selection_diagnostics,
                    fallback_category="score_file_no_pair_overlap",
                    fallback_reason=(
                        f"Score file {scores_path.name} had no overlapping pair keys for dataset {bundle.dataset} "
                        f"and split {split}."
                    ),
                )
                if fallback is not None:
                    return fallback
            fallback = _fallback_selection_from_heuristics(
                bundle,
                pairs,
                label=label,
                selector=selector,
                run_name=str(run_name),
                bench_method=str(bench_method),
                selection_diagnostics=selection_diagnostics,
                fallback_category="score_file_missing",
                fallback_reason=f"Score file is missing: {scores_path}",
            )
            if fallback is not None:
                return fallback
        fallback_reason = (
            f"No benchmark run and raw method could be resolved for dataset {bundle.dataset} split {split}."
            if not run_name and not bench_method
            else f"Incomplete benchmark discovery for dataset {bundle.dataset} split {split}: run={run_name!r}, method={bench_method!r}."
        )
        fallback = _fallback_selection_from_heuristics(
            bundle,
            pairs,
            label=label,
            selector=selector,
            run_name=str(run_name) if run_name is not None else None,
            bench_method=str(bench_method) if bench_method is not None else None,
            selection_diagnostics=selection_diagnostics,
            fallback_category="benchmark_resolution_missing",
            fallback_reason=fallback_reason,
        )
        if fallback is not None:
            return fallback

    if selector == "heuristic_hard_positive":
        pairs["score"] = pairs.apply(lambda r: _heuristic_challenge_score(r, bundle), axis=1)
        pairs = _sort_pairs_for_selection(pairs, ascending=False, tie_cols=["path_a", "path_b"])
        row = pairs.iloc[0]
        return SelectionResult(
            row=row,
            score=float(row["score"]),
            benchmark_context=None,
            selection_diagnostics={
                "selector": selector,
                "selection_driver": "heuristic_only",
                "benchmark_discovery_outcome": "not_requested",
                "benchmark_best_source": None,
                "benchmark_selection_status": "heuristic_only",
                "fallback_reason": None,
                "fallback_category": None,
                "chosen_run": None,
                "chosen_raw_benchmark_method": None,
                "chosen_canonical_method": None,
                "score_artifact_path_used": None,
                "benchmark_score": float(row["score"]),
                "benchmark_pair_match_signature_depth": None,
                "benchmark_backed_selection": False,
                "heuristic_fallback_used": False,
            },
            selection_policy="heuristic challenge ranking",
            selection_reason="Selected as the most challenging same-subject pair based on modality/session/impression diversity.",
        )
    if selector == "heuristic_hard_negative":
        pairs["score"] = pairs.apply(lambda r: _heuristic_negative_difficulty(r, bundle), axis=1)
        pairs = _sort_pairs_for_selection(pairs, ascending=False, tie_cols=["path_a", "path_b"])
        row = pairs.iloc[0]
        return SelectionResult(
            row=row,
            score=float(row["score"]),
            benchmark_context=None,
            selection_diagnostics={
                "selector": selector,
                "selection_driver": "heuristic_only",
                "benchmark_discovery_outcome": "not_requested",
                "benchmark_best_source": None,
                "benchmark_selection_status": "heuristic_only",
                "fallback_reason": None,
                "fallback_category": None,
                "chosen_run": None,
                "chosen_raw_benchmark_method": None,
                "chosen_canonical_method": None,
                "score_artifact_path_used": None,
                "benchmark_score": float(row["score"]),
                "benchmark_pair_match_signature_depth": None,
                "benchmark_backed_selection": False,
                "heuristic_fallback_used": False,
            },
            selection_policy="heuristic impostor difficulty ranking",
            selection_reason="Selected as a likely confusing impostor via subject-distance and capture similarity heuristics.",
        )

    return None



def resolve_benchmark_best_for_dataset_split(bundle: DatasetBundle, split: str) -> Dict[str, Any]:
    return dict(bundle.benchmark_best.get(f"{bundle.dataset}:{split}") or {})



def _benchmark_run_preference_key(dataset: str, run_name: str, *, validated: bool) -> Tuple[Any, ...]:
    showcase_runs = BENCHMARK_SHOWCASE_RUNS.get(dataset, ())
    if run_name in showcase_runs:
        return (0, showcase_runs.index(run_name), 0 if validated else 1, run_name)

    is_general_run = validated or run_name == "current" or run_name.startswith("current") or run_name.startswith("full_")
    run_kind_rank = 0 if run_name == "current" or run_name.startswith("current") else (1 if run_name.startswith("full_") else 2)
    return (1 if is_general_run else 2, 0 if validated else 1, run_kind_rank, run_name)



def _benchmark_candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        _benchmark_run_preference_key(
            str(candidate["dataset"]),
            str(candidate["run"]),
            validated=bool(candidate.get("validated", False)),
        ),
        0 if candidate.get("scores_csv_exists") else 1,
        str(candidate["method"]),
        str(candidate["run"]),
    )



def _better_metric_candidate(metric: str, candidate: Dict[str, Any], current: Optional[Dict[str, Any]]) -> bool:
    if current is None:
        return True

    candidate_value = candidate.get(metric)
    current_value = current.get(metric)
    if candidate_value is None:
        return False
    if current_value is None:
        return True

    if metric == "auc":
        if float(candidate_value) != float(current_value):
            return float(candidate_value) > float(current_value)
    else:
        if float(candidate_value) != float(current_value):
            return float(candidate_value) < float(current_value)

    return _benchmark_candidate_sort_key(candidate) < _benchmark_candidate_sort_key(current)



def _resolve_benchmark_artifact_path(run_dir: Path, raw_path: Any, fallback: Path) -> Path:
    text = str(raw_path or "").strip()
    if text:
        candidate = Path(text)
        if candidate.exists():
            return candidate
        if not candidate.is_absolute():
            candidate_in_run = run_dir / candidate
            if candidate_in_run.exists():
                return candidate_in_run
    return fallback



def _infer_benchmark_dataset(run_name: str, summary_row: Dict[str, Any], run_manifest: Dict[str, Any]) -> Optional[str]:
    config = _parse_json_text(summary_row.get("config_json"))
    dataset_candidates = [
        _normalize_dataset_key(summary_row.get("dataset")),
        _normalize_dataset_key(config.get("dataset")),
        _normalize_dataset_key(run_manifest.get("dataset")),
        _normalize_dataset_key(run_manifest.get("dataset", {}).get("name") if isinstance(run_manifest.get("dataset"), dict) else None),
    ]
    dataset_candidates.extend(dataset for dataset in DATASET_META if dataset in run_name.lower())
    for dataset in dataset_candidates:
        if dataset:
            return dataset
    return None



def _build_discovered_metric_entry(
    *,
    candidate: Dict[str, Any],
    metric: str,
    value_key: str,
) -> Dict[str, Any]:
    payload = {
        "method": candidate["method"],
        "run": candidate["run"],
        value_key: candidate.get(metric),
        "benchmark_best_source": "results_summary_scan",
        "summary_csv": candidate.get("summary_csv"),
    }
    if candidate.get("scores_csv"):
        payload["scores_csv"] = candidate.get("scores_csv")
    if candidate.get("method_semantics_epoch"):
        payload["method_semantics_epoch"] = candidate.get("method_semantics_epoch")
    return payload



def discover_benchmark_best_from_artifacts() -> Dict[str, Dict[str, Any]]:
    if not BENCH_ROOT.exists():
        return {}

    best: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run_dir in sorted(BENCH_ROOT.iterdir(), key=lambda item: item.name):
        if not run_dir.is_dir():
            continue
        if any(token in run_dir.name.lower() for token in BENCHMARK_EXCLUDED_RUN_TOKENS):
            continue

        summary_csv = run_dir / "results_summary.csv"
        if not summary_csv.exists():
            continue

        try:
            summary = pd.read_csv(summary_csv)
        except Exception:
            continue
        if summary.empty:
            continue

        run_manifest = _safe_read_json(run_dir / "run_manifest.json")
        validated = (run_dir / "validation.ok").exists()
        for _, row in summary.iterrows():
            raw = row.to_dict()
            dataset = _infer_benchmark_dataset(run_dir.name, raw, run_manifest)
            split = str(raw.get("split") or "").strip()
            method = str(raw.get("method") or "").strip()
            auc = _maybe_float(raw.get("auc"))
            eer = _maybe_float(raw.get("eer"))
            latency_reported = _maybe_float(raw.get("avg_ms_pair_reported"))
            latency_wall = _maybe_float(raw.get("avg_ms_pair_wall"))
            latency_ms = latency_reported if latency_reported is not None else latency_wall
            if not dataset or not split or not method or _canonical_verify_method_or_none(method) is None:
                continue
            if not benchmark_row_is_current(method, raw):
                continue

            scores_path = _resolve_benchmark_artifact_path(
                run_dir,
                raw.get("scores_csv"),
                run_dir / f"scores_{method}_{split}.csv",
            )
            candidate = {
                "dataset": dataset,
                "split": split,
                "run": run_dir.name,
                "method": method,
                "auc": auc,
                "eer": eer,
                "latency_ms": latency_ms,
                "validated": validated,
                "scores_csv": scores_path.name if scores_path.exists() else None,
                "scores_csv_exists": scores_path.exists(),
                "summary_csv": str(summary_csv.relative_to(ROOT).as_posix()) if summary_csv.exists() else None,
                "method_semantics_epoch": benchmark_row_semantics_epoch(method, raw),
            }
            key = f"{dataset}:{split}"
            resolved = best.setdefault(key, {})
            if _better_metric_candidate("auc", candidate, resolved.get("best_auc")):
                resolved["best_auc"] = candidate
            if _better_metric_candidate("eer", candidate, resolved.get("best_eer")):
                resolved["best_eer"] = candidate
            if latency_ms is not None and _better_metric_candidate("latency_ms", candidate, resolved.get("best_latency")):
                resolved["best_latency"] = candidate

    discovered: Dict[str, Dict[str, Any]] = {}
    for key, metrics in best.items():
        payload: Dict[str, Any] = {"benchmark_best_source": "results_summary_scan"}
        if metrics.get("best_auc"):
            payload["best_auc"] = _build_discovered_metric_entry(candidate=metrics["best_auc"], metric="auc", value_key="auc")
        if metrics.get("best_eer"):
            payload["best_eer"] = _build_discovered_metric_entry(candidate=metrics["best_eer"], metric="eer", value_key="eer")
        if metrics.get("best_latency"):
            payload["best_latency"] = _build_discovered_metric_entry(candidate=metrics["best_latency"], metric="latency_ms", value_key="latency_ms")
        discovered[key] = payload
    return discovered



def _filter_current_benchmark_best(payload: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    filtered: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        entry = {
            name: field
            for name, field in value.items()
            if name not in {"best_auc", "best_eer", "best_latency"}
        }
        kept_metric = False
        for metric_name in ("best_auc", "best_eer", "best_latency"):
            metric_payload = value.get(metric_name)
            if not isinstance(metric_payload, dict):
                continue
            method = str(metric_payload.get("method") or "").strip()
            if not method or not benchmark_row_is_current(method, metric_payload):
                continue
            entry[metric_name] = dict(metric_payload)
            kept_metric = True
        if kept_metric:
            filtered[key] = entry
    return filtered



def _annotate_benchmark_best_source(payload: Dict[str, Dict[str, Any]], source: str) -> Dict[str, Dict[str, Any]]:
    annotated: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        entry = dict(value)
        entry.setdefault("benchmark_best_source", source)
        for metric_name in ("best_auc", "best_eer", "best_latency"):
            metric_payload = entry.get(metric_name)
            if isinstance(metric_payload, dict):
                metric_entry = dict(metric_payload)
                metric_entry.setdefault("benchmark_best_source", source)
                entry[metric_name] = metric_entry
        annotated[key] = entry
    return annotated



def _merge_metric_entry(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    primary_method = str(primary.get("method") or "").strip()
    fallback_method = str(fallback.get("method") or "").strip()
    if primary_method and _canonical_verify_method_or_none(primary_method) is None:
        return dict(fallback) if fallback_method else dict(primary)
    if primary_method and fallback_method and primary_method != fallback_method:
        return dict(primary)
    merged = dict(fallback)
    merged.update({key: value for key, value in primary.items() if value is not None and value != ""})
    return merged



def _merge_benchmark_best(primary: Dict[str, Dict[str, Any]], fallback: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {key: dict(value) for key, value in fallback.items()}
    for key, value in primary.items():
        if key not in merged:
            merged[key] = dict(value)
            continue
        combined = dict(merged[key])
        for metric_name in ("best_auc", "best_eer", "best_latency"):
            primary_metric = value.get(metric_name)
            fallback_metric = combined.get(metric_name)
            if isinstance(primary_metric, dict) and isinstance(fallback_metric, dict):
                combined[metric_name] = _merge_metric_entry(primary_metric, fallback_metric)
            elif isinstance(primary_metric, dict):
                combined[metric_name] = dict(primary_metric)
        combined.update({name: field for name, field in value.items() if name not in {"best_auc", "best_eer", "best_latency"}})
        merged[key] = combined
    return merged



def load_benchmark_best() -> Dict[str, Dict[str, Any]]:
    discovered = _filter_current_benchmark_best(discover_benchmark_best_from_artifacts())
    if BEST_METHODS_JSON.exists():
        file_payload = _safe_read_json(BEST_METHODS_JSON)
        if file_payload:
            return _merge_benchmark_best(
                _filter_current_benchmark_best(_annotate_benchmark_best_source(file_payload, "best_methods_json")),
                discovered,
            )
    return discovered



def load_dataset_bundle(dataset: str, benchmark_best: Dict[str, Dict[str, Any]]) -> DatasetBundle:
    root = MANIFESTS_ROOT / dataset
    manifest = pd.read_csv(root / "manifest.csv")
    pairs_by_split = {
        split: pd.read_csv(root / f"pairs_{split}.csv")
        for split in ("train", "val", "test")
        if (root / f"pairs_{split}.csv").exists()
    }
    stats = _read_json(root / "stats.json")
    split_meta = _read_json(root / "split.json")
    protocol_note_path = str((root / "protocol_note.md").relative_to(ROOT).as_posix()) if (root / "protocol_note.md").exists() else None
    manifest_lookup = {str(row["path"]): row.to_dict() for _, row in manifest.iterrows()}
    manifest_lookup_by_signature = {path_signature(str(row["path"])): row.to_dict() for _, row in manifest.iterrows()}
    return DatasetBundle(
        dataset=dataset,
        manifest=manifest,
        pairs_by_split=pairs_by_split,
        stats=stats,
        split_meta=split_meta,
        protocol_note_path=protocol_note_path,
        manifest_lookup=manifest_lookup,
        manifest_lookup_by_signature=manifest_lookup_by_signature,
        benchmark_best=benchmark_best,
    )



def resolve_manifest_row(bundle: DatasetBundle, path_str: str) -> Dict[str, Any]:
    row = bundle.manifest_lookup.get(str(path_str))
    if row:
        return row
    sig = path_signature(str(path_str))
    row = bundle.manifest_lookup_by_signature.get(sig)
    if row:
        return row
    detail = availability_detail_for_path(path_str, traceable_to_manifest=False)
    return {
        "dataset": bundle.dataset,
        "capture": "unknown",
        "subject_id": -1,
        "frgp": 0,
        "path": str(path_str),
        "split": "unknown",
        "impression": None,
        "session": None,
        "source_modality": None,
        "availability_status": detail.status,
    }



def build_verify_case(bundle: DatasetBundle, plan: Dict[str, Any]) -> Optional[VerifyCase]:
    selected = select_case_row(bundle, plan)
    if selected is None:
        return None
    row = selected.row
    row_a = resolve_manifest_row(bundle, str(row["path_a"]))
    row_b = resolve_manifest_row(bundle, str(row["path_b"]))
    method, method_context = recommended_method_for_dataset(bundle, str(plan["split"]))
    modality_relation = infer_modality_relation(row_a, row_b, bundle.dataset)
    asset_a = build_asset(bundle.dataset, row_a, recommended_usage="verify_left")
    asset_b = build_asset(bundle.dataset, row_b, recommended_usage="verify_right")
    overall_availability = "available" if asset_a.availability_status == asset_b.availability_status == "available" else (
        "traceable_only" if asset_a.availability_status != "missing" and asset_b.availability_status != "missing" else "missing"
    )
    difficulty = str(plan["difficulty"])
    ground_truth = "match" if int(plan["label"]) == 1 else "non_match"
    case_type = str(plan["case_type"])
    dataset_label = DATASET_META[bundle.dataset]["label"]
    finger_a = asset_a.finger
    finger_b = asset_b.finger
    case_id = str(plan.get("case_id") or stable_id("case", bundle.dataset, plan["split"], case_type, row.get("pair_id", "na"), asset_a.signature, asset_b.signature))
    title = f"{dataset_label} • {case_type.replace('_', ' ')}"
    description = (
        f"Curated {ground_truth} case from {dataset_label} ({plan['split']}) selected for {difficulty} demonstration coverage."
    )
    benchmark_context = dict(selected.benchmark_context or {})
    benchmark_context.update(method_context)
    benchmark_context = normalize_benchmark_context(benchmark_context, split=str(plan["split"]))
    selection_diagnostics = dict(selected.selection_diagnostics or {})
    selection_diagnostics["chosen_canonical_method"] = (
        benchmark_context.get("canonical_method")
        or selection_diagnostics.get("chosen_canonical_method")
        or method
    )
    selection_diagnostics["chosen_raw_benchmark_method"] = (
        benchmark_context.get("benchmark_method")
        or selection_diagnostics.get("chosen_raw_benchmark_method")
    )
    selection_diagnostics["chosen_run"] = (
        benchmark_context.get("benchmark_run")
        or benchmark_context.get("run")
        or selection_diagnostics.get("chosen_run")
    )
    selection_diagnostics["score_artifact_path_used"] = (
        benchmark_context.get("score_artifact_path_used")
        or selection_diagnostics.get("score_artifact_path_used")
    )
    selection_diagnostics["benchmark_best_source"] = (
        benchmark_context.get("benchmark_best_source")
        or selection_diagnostics.get("benchmark_best_source")
    )
    availability_detail = {
        "path_a": asset_a.availability_detail.model_dump(),
        "path_b": asset_b.availability_detail.model_dump(),
        "all_assets_locally_available": overall_availability == "available",
    }
    is_demo_safe = bool(plan.get("demo_safe", False))
    tags = [bundle.dataset, case_type, difficulty, modality_relation, ground_truth]
    if bundle.dataset in {"polyu_cross", "unsw_2d3d"}:
        tags.append("cross_modality")
    if bundle.dataset in {"l3_sf_v2", "polyu_3d"}:
        tags.append("challenging")
    return VerifyCase(
        case_id=case_id,
        title=title,
        description=description,
        dataset=bundle.dataset,
        split=str(plan["split"]),
        case_type=case_type,
        difficulty=difficulty,
        ground_truth=ground_truth,
        image_a=asset_a,
        image_b=asset_b,
        subject_a=int(row.get("subject_a", row_a.get("subject_id", -1))),
        subject_b=int(row.get("subject_b", row_b.get("subject_id", -1))),
        finger_a=finger_a,
        finger_b=finger_b,
        capture_a=asset_a.capture,
        capture_b=asset_b.capture,
        modality_relation=modality_relation,
        source_pair_file=f"data/manifests/{bundle.dataset}/pairs_{plan['split']}.csv",
        source_pair_row_id=str(int(row.get("pair_id"))) if pd.notna(row.get("pair_id")) else stable_id("pairrow", bundle.dataset, plan["split"], asset_a.signature, asset_b.signature),
        recommended_method=method,
        tags=sorted(dict.fromkeys(tags)),
        is_demo_safe=is_demo_safe,
        availability_status=overall_availability,
        availability_detail=availability_detail,
        selection_reason=selected.selection_reason,
        selection_policy=selected.selection_policy,
        benchmark_context=benchmark_context,
        selection_diagnostics=selection_diagnostics,
    )





def _safe_sort_columns(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    return [col for col in preferred if col in df.columns]


def _row_session_int(row: Dict[str, Any]) -> Optional[int]:
    value = row.get("session")
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _canonical_identity_rows(bundle: DatasetBundle, manifest: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = (manifest.copy() if manifest is not None else bundle.manifest.copy()).reset_index(drop=True)
    if rows.empty:
        return rows
    if bundle.dataset in {"nist_sd300b", "nist_sd300c"}:
        return rows[rows["capture"].isin(["plain", "roll"])].copy().reset_index(drop=True)
    if bundle.dataset == "polyu_cross":
        return rows[rows["capture"].isin(["contact_based", "contactless"])].copy().reset_index(drop=True)
    if bundle.dataset == "unsw_2d3d":
        return rows[rows["source_modality"].isin(["optical_2d", "reconstructed_3d"])].copy().reset_index(drop=True)
    if bundle.dataset == "polyu_3d":
        return rows[rows["source_modality"] == "contactless_3d_surface"].copy().reset_index(drop=True)
    if bundle.dataset == "l3_sf_v2":
        return rows[rows["source_modality"] == "synthetic_level3"].copy().reset_index(drop=True)
    return rows


def _row_matches_identity_role(dataset: str, row: Dict[str, Any], role: Literal["enrollment", "probe"]) -> bool:
    capture = str(row.get("capture", "unknown"))
    source_modality = str(row.get("source_modality", capture))
    session = _row_session_int(row)
    if dataset in {"nist_sd300b", "nist_sd300c"}:
        return capture == ("plain" if role == "enrollment" else "roll")
    if dataset == "polyu_cross":
        return capture == ("contact_based" if role == "enrollment" else "contactless")
    if dataset == "unsw_2d3d":
        return source_modality == ("optical_2d" if role == "enrollment" else "reconstructed_3d")
    if dataset == "polyu_3d":
        if source_modality != "contactless_3d_surface":
            return False
        return session == (1 if role == "enrollment" else 2)
    if dataset == "l3_sf_v2":
        return source_modality == "synthetic_level3"
    return False


def _subject_supports_identity_flow(bundle: DatasetBundle, subject_rows: pd.DataFrame) -> bool:
    if subject_rows.empty:
        return False
    if bundle.dataset == "l3_sf_v2":
        return len(subject_rows) >= 2
    subject_dicts = [row.to_dict() for _, row in subject_rows.iterrows()]
    has_enrollment = any(_row_matches_identity_role(bundle.dataset, row, "enrollment") for row in subject_dicts)
    has_probe = any(_row_matches_identity_role(bundle.dataset, row, "probe") for row in subject_dicts)
    return bool(has_enrollment and has_probe)


def first_subject_with_enough_assets(bundle: DatasetBundle) -> Tuple[int, pd.DataFrame]:
    manifest = _canonical_identity_rows(bundle)
    if manifest.empty:
        raise ValueError(f"{bundle.dataset} has no canonical identity-flow assets in manifest.csv")
    split_pref = {"val": 0, "test": 1, "train": 2}
    manifest["split_rank"] = manifest["split"].map(split_pref).fillna(99)
    subject_counts = (
        manifest.groupby(["subject_id", "split_rank", "split"]).size().reset_index(name="n").sort_values(["split_rank", "subject_id"])
    )
    for _, row in subject_counts.iterrows():
        subject_id = int(row["subject_id"])
        subset = manifest[manifest["subject_id"] == subject_id].copy()
        if _subject_supports_identity_flow(bundle, subset):
            sort_cols = _safe_sort_columns(subset, ["split_rank", "capture", "session", "frgp", "impression", "path"])
            return subject_id, subset.sort_values(sort_cols, kind="mergesort") if sort_cols else subset
    raise ValueError(f"{bundle.dataset} has no subject with canonical enrollment/probe coverage")



def exemplar_usage(bundle: DatasetBundle, row: Dict[str, Any]) -> Tuple[str, str]:
    capture = str(row.get("capture", "unknown"))
    source_modality = str(row.get("source_modality", capture))
    session = row.get("session")
    ppi = row.get("ppi")
    ppi_hint = f"{int(ppi)}ppi" if pd.notna(ppi) and int(ppi) > 0 else source_modality
    if bundle.dataset in {"nist_sd300b", "nist_sd300c"}:
        usage = "recommended_enrollment" if capture == "plain" else "recommended_probe"
    elif bundle.dataset == "polyu_cross":
        usage = "recommended_enrollment" if capture == "contact_based" else "recommended_probe"
    elif bundle.dataset == "unsw_2d3d":
        usage = "recommended_enrollment" if source_modality == "optical_2d" else "recommended_probe"
    elif bundle.dataset == "polyu_3d":
        if source_modality == "contactless_3d_surface":
            usage = "recommended_enrollment" if int(session or 0) == 1 else "recommended_probe"
            ppi_hint = f"surface_session_{int(session or 0)}"
        else:
            usage = "auxiliary"
    elif bundle.dataset == "l3_sf_v2":
        usage = "recommended_enrollment"
    else:
        usage = "recommended_probe" if "3d" in source_modality else "recommended_enrollment"
    return usage, ppi_hint



def pick_identity_exemplars(bundle: DatasetBundle, subject_rows: pd.DataFrame, limit: int = 6) -> List[CatalogAsset]:
    subject_rows = _canonical_identity_rows(bundle, subject_rows)
    chosen: List[CatalogAsset] = []
    seen_keys: set[Tuple[Any, ...]] = set()
    seen_asset_ids: set[str] = set()

    sort_cols = _safe_sort_columns(subject_rows, ["split_rank", "capture", "session", "frgp", "impression", "path"])
    ordered = subject_rows.sort_values(sort_cols, kind="mergesort") if sort_cols else subject_rows.copy()

    def _append_rows(rows: pd.DataFrame, *, forced_usage: Optional[str] = None, max_to_add: Optional[int] = None) -> None:
        added = 0
        for _, row in rows.iterrows():
            if len(chosen) >= limit:
                return
            data = row.to_dict()
            usage, qhint = exemplar_usage(bundle, data)
            if forced_usage is not None:
                usage = forced_usage
            if usage == "auxiliary":
                continue
            key = (
                data.get("split"),
                data.get("capture"),
                data.get("source_modality"),
                data.get("session"),
                data.get("frgp"),
            )
            if key in seen_keys and len(chosen) >= 2:
                continue
            asset = build_asset(bundle.dataset, data, recommended_usage=usage, quality_hint=qhint)
            if asset.asset_id in seen_asset_ids:
                continue
            chosen.append(asset)
            seen_keys.add(key)
            seen_asset_ids.add(asset.asset_id)
            added += 1
            if max_to_add is not None and added >= max_to_add:
                return

    if bundle.dataset != "l3_sf_v2":
        enrollment_rows = ordered[[ _row_matches_identity_role(bundle.dataset, row.to_dict(), "enrollment") for _, row in ordered.iterrows() ]]
        probe_rows = ordered[[ _row_matches_identity_role(bundle.dataset, row.to_dict(), "probe") for _, row in ordered.iterrows() ]]
        _append_rows(enrollment_rows, forced_usage="recommended_enrollment", max_to_add=1)
        _append_rows(probe_rows, forced_usage="recommended_probe", max_to_add=1)
    _append_rows(ordered)

    if len(chosen) < 2:
        _append_rows(ordered.head(limit))

    dedup: Dict[str, CatalogAsset] = {}
    for asset in chosen:
        dedup.setdefault(asset.asset_id, asset)
    return list(dedup.values())



def build_identity_record(bundle: DatasetBundle) -> IdentityRecord:
    subject_id, subject_rows = first_subject_with_enough_assets(bundle)
    exemplars = pick_identity_exemplars(bundle, subject_rows, limit=6)
    enrollment = [asset.asset_id for asset in exemplars if asset.recommended_usage == "recommended_enrollment"] or [exemplars[0].asset_id]
    probes = [asset.asset_id for asset in exemplars if asset.recommended_usage == "recommended_probe"] or [exemplars[-1].asset_id]
    gallery_role = "difficult" if bundle.dataset in {"polyu_cross", "unsw_2d3d", "polyu_3d"} else "standard"
    identity_id = stable_id("identity", bundle.dataset, subject_id)
    return IdentityRecord(
        identity_id=identity_id,
        dataset=bundle.dataset,
        display_name=f"{DATASET_META[bundle.dataset]['label']} Subject {subject_id}",
        subject_id=subject_id,
        gallery_role=gallery_role,
        enrollment_candidates=enrollment,
        probe_candidates=probes,
        recommended_enrollment_asset_id=enrollment[0],
        recommended_probe_asset_id=probes[0],
        tags=sorted(dict.fromkeys([bundle.dataset, gallery_role, DATASET_META[bundle.dataset]["protocol"]])),
        is_demo_safe=bundle.dataset in IDENTITY_DEMO_SAFE_DATASETS,
        exemplars=exemplars,
    )



def find_asset(identity: IdentityRecord, asset_id: str) -> CatalogAsset:
    for asset in identity.exemplars:
        if asset.asset_id == asset_id:
            return asset
    raise KeyError(asset_id)



def build_identification_scenarios(bundles: Dict[str, DatasetBundle], identities: List[IdentityRecord]) -> List[IdentificationScenario]:
    identities_by_dataset = {identity.dataset: identity for identity in identities}
    scenarios: List[IdentificationScenario] = []

    # Positive identification from NIST.
    nist = identities_by_dataset.get("nist_sd300b")
    if nist and nist.recommended_probe_asset_id:
        probe = find_asset(nist, nist.recommended_probe_asset_id)
        method, _ = recommended_method_for_dataset(bundles[nist.dataset], probe.split)
        scenarios.append(
            IdentificationScenario(
                scenario_id=stable_id("scenario", nist.dataset, "positive", nist.identity_id, probe.asset_id),
                scenario_type="positive_identification",
                dataset=nist.dataset,
                title="Positive identification from the curated NIST gallery",
                description="Enroll a plain reference and probe with the corresponding roll image from the same subject.",
                enrollment_identity_ids=[nist.identity_id],
                probe_asset=probe,
                expected_identity_id=nist.identity_id,
                difficulty="easy",
                recommended_method=method,
                tags=[nist.dataset, "positive_identification", "plain_to_roll"],
                is_demo_safe=nist.is_demo_safe,
            )
        )

    # Difficult identification from PolyU Cross.
    polyu = identities_by_dataset.get("polyu_cross")
    if polyu and polyu.recommended_probe_asset_id:
        probe = find_asset(polyu, polyu.recommended_probe_asset_id)
        method, _ = recommended_method_for_dataset(bundles[polyu.dataset], probe.split)
        scenarios.append(
            IdentificationScenario(
                scenario_id=stable_id("scenario", polyu.dataset, "difficult", polyu.identity_id, probe.asset_id),
                scenario_type="difficult_identification",
                dataset=polyu.dataset,
                title="Cross-modality difficult identification",
                description="Enroll with one modality and probe with the paired contactless/contact-based counterpart for a hard 1:N story.",
                enrollment_identity_ids=[polyu.identity_id],
                probe_asset=probe,
                expected_identity_id=polyu.identity_id,
                difficulty="hard",
                recommended_method=method,
                tags=[polyu.dataset, "cross_modality", "difficult_identification"],
                is_demo_safe=polyu.is_demo_safe,
            )
        )

    # No-match scenario using UNSW probe from a different subject than the enrolled identity.
    unsw_bundle = bundles.get("unsw_2d3d")
    unsw_identity = identities_by_dataset.get("unsw_2d3d")
    if unsw_bundle and unsw_identity:
        probe_row = None
        subject = unsw_identity.subject_id
        manifest = _canonical_identity_rows(unsw_bundle)
        manifest = manifest[
            (manifest["subject_id"] != subject)
            & (manifest["split"] == subject_rows_preferred_split(unsw_identity))
            & (manifest["source_modality"] == "reconstructed_3d")
        ]
        if manifest.empty:
            manifest = _canonical_identity_rows(unsw_bundle)
            manifest = manifest[
                (manifest["subject_id"] != subject)
                & (manifest["source_modality"] == "reconstructed_3d")
            ]
        if not manifest.empty:
            sort_cols = _safe_sort_columns(manifest, ["subject_id", "capture", "session", "frgp", "impression", "path"])
            probe_row = (manifest.sort_values(sort_cols, kind="mergesort") if sort_cols else manifest).iloc[0].to_dict()
        if probe_row is not None:
            probe = build_asset(unsw_bundle.dataset, probe_row, recommended_usage="recommended_probe")
            method, _ = recommended_method_for_dataset(unsw_bundle, probe.split)
            scenarios.append(
                IdentificationScenario(
                    scenario_id=stable_id("scenario", unsw_bundle.dataset, "no_match", unsw_identity.identity_id, probe.asset_id),
                    scenario_type="no_match",
                    dataset=unsw_bundle.dataset,
                    title="No-match probe against the curated gallery",
                    description="Probe with a different subject to exercise a low-confidence or no-match 1:N path.",
                    enrollment_identity_ids=[unsw_identity.identity_id],
                    probe_asset=probe,
                    expected_identity_id=None,
                    difficulty="challenging",
                    recommended_method=method,
                    tags=[unsw_bundle.dataset, "no_match", "cross_modality"],
                    is_demo_safe=unsw_identity.is_demo_safe,
                )
            )
    return scenarios



def subject_rows_preferred_split(identity: IdentityRecord) -> str:
    exemplar_splits = [asset.split for asset in identity.exemplars if asset.split]
    pref_order = ["val", "test", "train"]
    for split in pref_order:
        if split in exemplar_splits:
            return split
    return exemplar_splits[0] if exemplar_splits else "train"



def build_browser_seed(bundle: DatasetBundle, limit: int = 6) -> BrowserSeedEntry:
    manifest = _canonical_identity_rows(bundle)
    if manifest.empty:
        return BrowserSeedEntry(
            dataset=bundle.dataset,
            items=[],
            selection_policy="Stable greedy coverage over canonical identity-flow assets only.",
            coverage_summary={"n_items": 0, "subjects": [], "captures": [], "splits": [], "availability_breakdown": {}},
        )
    manifest["split_rank"] = manifest["split"].map({"val": 0, "test": 1, "train": 2}).fillna(99)
    sort_cols = _safe_sort_columns(manifest, ["split_rank", "subject_id", "capture", "session", "frgp", "impression", "path"])
    manifest = manifest.sort_values(sort_cols, kind="mergesort") if sort_cols else manifest
    items: List[CatalogAsset] = []
    seen_subjects: set[int] = set()
    seen_keys: set[Tuple[Any, ...]] = set()
    for _, row in manifest.iterrows():
        data = row.to_dict()
        key = (data.get("split"), data.get("capture"), data.get("subject_id"), data.get("frgp"))
        if key in seen_keys:
            continue
        if len(seen_subjects) >= max(2, limit // 2) and data.get("subject_id") not in seen_subjects:
            continue
        usage, qhint = exemplar_usage(bundle, data)
        if usage == "auxiliary":
            continue
        items.append(build_asset(bundle.dataset, data, recommended_usage=usage, quality_hint=qhint))
        seen_subjects.add(int(data.get("subject_id")))
        seen_keys.add(key)
        if len(items) >= limit:
            break
    if len(items) < limit:
        for _, row in manifest.head(limit).iterrows():
            usage, qhint = exemplar_usage(bundle, row.to_dict())
            if usage == "auxiliary":
                continue
            asset = build_asset(bundle.dataset, row.to_dict(), recommended_usage=usage, quality_hint=qhint)
            if asset.asset_id not in {i.asset_id for i in items}:
                items.append(asset)
            if len(items) >= limit:
                break
    coverage = {
        "n_items": len(items),
        "subjects": sorted({int(item.subject_id) for item in items})[:10],
        "captures": sorted({item.capture for item in items}),
        "splits": sorted({item.split for item in items}),
        "availability_breakdown": pd.Series([item.availability_status for item in items]).value_counts().to_dict(),
    }
    return BrowserSeedEntry(
        dataset=bundle.dataset,
        items=items,
        selection_policy="Stable greedy coverage over canonical split, subject, capture and finger diversity.",
        coverage_summary=coverage,
    )



def _case_selection_diagnostics_payload(case: VerifyCase) -> Dict[str, Any]:
    payload = dict(case.selection_diagnostics or {})
    payload.update(
        {
            "case_id": case.case_id,
            "dataset": case.dataset,
            "split": case.split,
            "selection_policy": case.selection_policy,
        }
    )
    return payload


def build_verify_selection_diagnostics(
    bundles: Dict[str, DatasetBundle],
    verify_cases: List[VerifyCase],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    plans_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for plan in VERIFY_CASE_PLANS:
        plans_by_key.setdefault((str(plan["dataset"]), str(plan["split"])), []).append(plan)

    cases_by_key: Dict[Tuple[str, str], List[VerifyCase]] = {}
    for case in verify_cases:
        cases_by_key.setdefault((case.dataset, case.split), []).append(case)

    summaries_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    case_driver_counts: Counter[str] = Counter()
    degraded_dataset_keys: set[str] = set()
    missing_evidence_keys: set[str] = set()
    degraded_case_ids: set[str] = set()

    for (dataset, split), plans in sorted(plans_by_key.items()):
        bundle = bundles[dataset]
        best = resolve_benchmark_best_for_dataset_split(bundle, split)
        best_auc = dict(best.get("best_auc") or {})
        best_source = best_auc.get("benchmark_best_source") or best.get("benchmark_best_source")
        best_raw_method = str(best_auc.get("method") or "").strip() or None
        best_canonical_method = _canonical_verify_method_or_none(best_raw_method)
        best_run = str(best_auc.get("run") or "").strip() or None
        planned_case_ids = [str(plan["case_id"]) for plan in plans]
        cases = cases_by_key.get((dataset, split), [])
        case_payloads = [_case_selection_diagnostics_payload(case) for case in cases]

        driver_counts: Counter[str] = Counter()
        fallback_counts: Counter[str] = Counter()
        selection_status_counts: Counter[str] = Counter()
        for payload in case_payloads:
            driver = str(payload.get("selection_driver") or "unknown")
            driver_counts.update([driver])
            case_driver_counts.update([driver])
            selection_status_counts.update([str(payload.get("benchmark_selection_status") or "unknown")])
            fallback_category = payload.get("fallback_category")
            if fallback_category:
                fallback_counts.update([str(fallback_category)])
            if payload.get("heuristic_fallback_used"):
                degraded_case_ids.add(str(payload["case_id"]))

        benchmark_best_available = bool(best_auc)
        benchmark_resolution_complete = bool(best_run and best_raw_method and best_canonical_method)
        benchmark_backed_selection_succeeded = selection_status_counts.get("benchmark_score_used", 0) > 0
        heuristic_fallback_used = driver_counts.get("heuristic_fallback", 0) > 0
        score_files_missing = selection_status_counts.get("score_file_missing", 0) > 0
        score_files_unparsable = selection_status_counts.get("score_file_unparsable", 0) > 0
        score_files_missing_columns = selection_status_counts.get("score_file_missing_columns", 0) > 0
        score_files_no_pair_overlap = selection_status_counts.get("score_file_no_pair_overlap", 0) > 0

        notes: List[str] = []
        dataset_split_key = f"{dataset}:{split}"
        if not benchmark_best_available:
            missing_evidence_keys.add(dataset_split_key)
            degraded_dataset_keys.add(dataset_split_key)
            notes.append(
                "No benchmark-best evidence was available for this curated verify selection group; benchmark selectors degraded to heuristics."
            )
        elif not benchmark_resolution_complete:
            degraded_dataset_keys.add(dataset_split_key)
            notes.append(
                "Benchmark-best evidence was present but incomplete, so curated verify selection could not stay fully benchmark-backed."
            )
        else:
            notes.append(
                f"Benchmark-best evidence was resolved from {best_source or 'unknown_source'}."
            )
        if benchmark_backed_selection_succeeded:
            notes.append(
                f"Benchmark-backed selection succeeded for {selection_status_counts.get('benchmark_score_used', 0)} curated case(s)."
            )
        if heuristic_fallback_used:
            degraded_dataset_keys.add(dataset_split_key)
            notes.append(
                f"Heuristic fallback was used for {driver_counts.get('heuristic_fallback', 0)} curated case(s)."
            )
        if score_files_missing:
            degraded_dataset_keys.add(dataset_split_key)
            notes.append("One or more benchmark score files were missing.")
        if score_files_unparsable:
            degraded_dataset_keys.add(dataset_split_key)
            notes.append("One or more benchmark score files could not be parsed.")
        if score_files_missing_columns:
            degraded_dataset_keys.add(dataset_split_key)
            notes.append("One or more benchmark score files were missing required selection columns.")
        if score_files_no_pair_overlap:
            degraded_dataset_keys.add(dataset_split_key)
            notes.append("One or more benchmark score files had no overlapping pair keys with the curated manifest pairs.")
        if len(cases) < len(plans):
            degraded_dataset_keys.add(dataset_split_key)
            notes.append(f"Built {len(cases)} of {len(plans)} planned curated verify case(s) successfully.")

        summary = {
            "dataset": dataset,
            "split": split,
            "planned_case_ids": planned_case_ids,
            "built_case_ids": [case.case_id for case in cases],
            "planned_curated_cases": len(plans),
            "curated_cases_built_successfully": len(cases),
            "benchmark_best_available": benchmark_best_available,
            "benchmark_best_source": best_source,
            "benchmark_best_run": best_run,
            "benchmark_best_raw_benchmark_method": best_raw_method,
            "benchmark_best_canonical_method": best_canonical_method,
            "benchmark_resolution_complete": benchmark_resolution_complete,
            "benchmark_backed_selection_succeeded": benchmark_backed_selection_succeeded,
            "heuristic_fallback_used": heuristic_fallback_used,
            "score_files_missing": score_files_missing,
            "score_files_unparsable": score_files_unparsable,
            "score_files_missing_columns": score_files_missing_columns,
            "score_files_no_pair_overlap": score_files_no_pair_overlap,
            "selection_driver_counts": dict(sorted(driver_counts.items())),
            "fallback_category_counts": dict(sorted(fallback_counts.items())),
            "benchmark_selection_status_counts": dict(sorted(selection_status_counts.items())),
            "notes": notes,
        }
        summaries_by_dataset.setdefault(dataset, []).append(summary)

    total_planned = len(VERIFY_CASE_PLANS)
    total_built = len(verify_cases)
    healthy = total_built == total_planned and not degraded_dataset_keys
    health_notes: List[str] = []
    if missing_evidence_keys:
        health_notes.append("Missing benchmark evidence for: " + ", ".join(sorted(missing_evidence_keys)))
    if degraded_dataset_keys:
        health_notes.append("Degraded curated verify selection for: " + ", ".join(sorted(degraded_dataset_keys)))
    if total_built != total_planned:
        health_notes.append(f"Built {total_built} of {total_planned} planned curated verify cases.")

    catalog_build_health = {
        "status": "healthy" if healthy else "degraded",
        "total_verify_cases_planned": total_planned,
        "total_verify_cases_built": total_built,
        "case_selection_driver_counts": dict(sorted(case_driver_counts.items())),
        "datasets_with_missing_benchmark_evidence": sorted(missing_evidence_keys),
        "datasets_with_degraded_selection": sorted(degraded_dataset_keys),
        "degraded_case_ids": sorted(degraded_case_ids),
        "notes": health_notes,
    }
    return summaries_by_dataset, catalog_build_health


def build_source_dataset(bundle: DatasetBundle, verify_cases: List[VerifyCase], identity_records: List[IdentityRecord], browser_seed: BrowserSeedEntry, verify_selection_diagnostics: Optional[List[Dict[str, Any]]] = None) -> SourceDataset:
    root = MANIFESTS_ROOT / bundle.dataset
    benchmark_runs: List[Dict[str, Any]] = []
    for key, payload in bundle.benchmark_best.items():
        if not key.startswith(f"{bundle.dataset}:"):
            continue
        split = key.split(":", 1)[1]
        best_auc = payload.get("best_auc") or {}
        if best_auc:
            benchmark_runs.append(
                {
                    "split": split,
                    "run": best_auc.get("run"),
                    "best_auc_method": best_auc.get("method"),
                    "benchmark_best_source": best_auc.get("benchmark_best_source") or payload.get("benchmark_best_source"),
                }
            )
    included = any(v.dataset == bundle.dataset for v in verify_cases) or any(i.dataset == bundle.dataset for i in identity_records) or browser_seed.dataset == bundle.dataset
    notes = [str(DATASET_META[bundle.dataset]["notes"])]
    if bundle.dataset not in VERIFY_DEMO_DATASETS:
        notes.append("Catalog-included for identification and browser coverage, but intentionally not exposed as one of the 7 curated verify demo cases.")
    stats_rows = int(bundle.stats.get("manifest_rows", len(bundle.manifest)))
    return SourceDataset(
        dataset=bundle.dataset,
        dataset_label=DATASET_META[bundle.dataset]["label"],
        manifest_path=f"data/manifests/{bundle.dataset}/manifest.csv",
        pair_files={split: f"data/manifests/{bundle.dataset}/pairs_{split}.csv" for split in sorted(bundle.pairs_by_split.keys())},
        stats_path=f"data/manifests/{bundle.dataset}/stats.json",
        split_path=f"data/manifests/{bundle.dataset}/split.json",
        protocol_note_path=bundle.protocol_note_path,
        benchmark_runs=benchmark_runs,
        manifest_rows=stats_rows,
        unique_subjects=int(bundle.stats.get("unique_subjects", bundle.manifest["subject_id"].nunique())),
        included_in_catalog=included,
        exclusion_reason=None if included else "No curated case/identity/browser item could be derived deterministically.",
        notes=notes,
        verify_selection_diagnostics=list(verify_selection_diagnostics or []),
    )



def materialized_asset_relative_path(asset: Dict[str, Any]) -> str:
    dataset = str(asset.get("dataset") or asset.get("traceability", {}).get("source_dataset") or "unknown")
    return f"data/samples/assets/{dataset}/{asset['asset_id']}{MATERIALIZED_ASSET_EXT}"


def materialized_asset_thumbnail_relative_path(asset: Dict[str, Any]) -> str:
    dataset = str(asset.get("dataset") or asset.get("traceability", {}).get("source_dataset") or "unknown")
    return f"data/samples/assets/{dataset}/thumbnails/{asset['asset_id']}{MATERIALIZED_ASSET_EXT}"


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


def _materialized_dimensions(image: np.ndarray) -> Dict[str, int]:
    height, width = image.shape[:2]
    return {"width": int(width), "height": int(height)}


def _read_materialization_source(path: Path) -> np.ndarray:
    return read_deterministic_preview_source(path)


def _resize_image_to_max_edge(image: np.ndarray, *, max_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image.copy()

    scale = float(max_edge) / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write PNG asset: {path}")


def _absolute_source_candidate(path_str: str) -> Optional[Path]:
    normalized = str(path_str or "").strip()
    if not normalized:
        return None
    try:
        candidate = Path(normalized)
    except Exception:
        return None
    if not candidate.is_absolute():
        return None
    return candidate.resolve() if candidate.exists() else None


def _resolve_source_local_path(asset: Dict[str, Any], source_path: str, source_relative_path: Optional[str]) -> Optional[Path]:
    candidates: List[Optional[Path]] = [
        _candidate_local_path(source_relative_path),
        _candidate_local_path(_extract_relative_data_path(source_path)),
        _absolute_source_candidate(source_path),
        _candidate_local_path(asset.get("source_relative_path")),
        _candidate_local_path(_extract_relative_data_path(str(asset.get("source_path") or ""))),
        _candidate_local_path(asset.get("relative_path")),
        _candidate_local_path(_extract_relative_data_path(str(asset.get("path") or ""))),
        _absolute_source_candidate(str(asset.get("path") or "")),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def materialize_asset_dict(asset: Dict[str, Any]) -> Dict[str, Any]:
    source_path = str(asset.get("source_path") or asset.get("path") or "")
    source_relative_path = asset.get("source_relative_path") or _extract_relative_data_path(source_path)
    source_local = _resolve_source_local_path(asset, source_path, source_relative_path)
    source_local_exists = bool(source_local and source_local.exists())
    materialized_relative_path = materialized_asset_relative_path(asset)
    thumbnail_relative_path = materialized_asset_thumbnail_relative_path(asset)
    materialized_abs_path = (ROOT / materialized_relative_path).resolve()
    thumbnail_abs_path = (ROOT / thumbnail_relative_path).resolve()
    rendered_dimensions: Optional[Dict[str, int]] = None
    thumbnail_dimensions: Optional[Dict[str, int]] = None
    render_error: Optional[str] = None

    if source_local_exists:
        try:
            rendered = _read_materialization_source(source_local)
            thumbnail = _resize_image_to_max_edge(rendered, max_edge=MATERIALIZED_ASSET_THUMBNAIL_SIZE)
            _write_png(materialized_abs_path, rendered)
            _write_png(thumbnail_abs_path, thumbnail)
            rendered_dimensions = _materialized_dimensions(rendered)
            thumbnail_dimensions = _materialized_dimensions(thumbnail)
        except Exception as exc:
            render_error = str(exc)

    updated = dict(asset)
    updated["path"] = materialized_relative_path
    updated["relative_path"] = materialized_relative_path
    updated["source_path"] = source_path
    updated["source_relative_path"] = source_relative_path
    updated["thumbnail_path"] = thumbnail_relative_path
    updated["content_type"] = "image/png"
    updated["dimensions"] = rendered_dimensions
    updated["thumbnail_dimensions"] = thumbnail_dimensions
    updated["availability_status"] = "available" if rendered_dimensions is not None else "missing"
    updated["materialized_asset_kind"] = MATERIALIZED_ASSET_KIND
    updated["availability_detail"] = {
        "status": "available" if rendered_dimensions is not None else "missing",
        "local_exists": materialized_abs_path.exists(),
        "traceable_to_manifest": True,
        "relative_path": materialized_relative_path,
        "resolved_local_path": str(materialized_abs_path),
        "source_relative_path": source_relative_path,
        "source_resolved_local_path": str(source_local) if source_local else None,
        "source_local_exists": source_local_exists,
    }
    if render_error:
        updated["availability_detail"]["render_error"] = render_error
    updated.setdefault("traceability", {})
    updated["traceability"]["materialized_asset_path"] = materialized_relative_path
    updated["traceability"]["materialized_thumbnail_path"] = thumbnail_relative_path
    updated["traceability"]["materialized_asset_kind"] = MATERIALIZED_ASSET_KIND
    return updated


def materialize_catalog_assets(payload: Dict[str, Any]) -> Dict[str, Any]:
    if ASSETS_ROOT.exists():
        _safe_rmtree(ASSETS_ROOT)
    ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
    materialized_cache: Dict[str, Dict[str, Any]] = {}

    def update_asset(asset: Dict[str, Any]) -> Dict[str, Any]:
        asset_id = str(asset["asset_id"])
        if asset_id not in materialized_cache:
            materialized_cache[asset_id] = materialize_asset_dict(asset)
        cached = materialized_cache[asset_id]
        merged = dict(asset)
        merged.update({k: v for k, v in cached.items() if k not in {"recommended_usage", "quality_hint", "thumbnail_hint"}})
        return merged

    for case in payload.get("verify_cases", []):
        case["image_a"] = update_asset(case["image_a"])
        case["image_b"] = update_asset(case["image_b"])
        both_available = case["image_a"]["availability_status"] == case["image_b"]["availability_status"] == "available"
        case["availability_status"] = "available" if both_available else "missing"
        case["availability_detail"] = {
            "path_a": case["image_a"]["availability_detail"],
            "path_b": case["image_b"]["availability_detail"],
            "all_assets_locally_available": both_available,
        }

    identify_gallery = payload.get("identify_gallery", {})
    for identity in identify_gallery.get("identities", []):
        identity["exemplars"] = [update_asset(asset) for asset in identity.get("exemplars", [])]
    for scenario in identify_gallery.get("demo_scenarios", []):
        scenario["probe_asset"] = update_asset(scenario["probe_asset"])

    for entry in payload.get("dataset_browser_seed", []):
        entry["items"] = [update_asset(asset) for asset in entry.get("items", [])]
        availability_breakdown = pd.Series([item["availability_status"] for item in entry.get("items", [])]).value_counts().to_dict() if entry.get("items") else {}
        entry.setdefault("coverage_summary", {})["availability_breakdown"] = availability_breakdown

    payload.setdefault("metadata", {})["materialized_asset_root"] = str(ASSETS_ROOT.relative_to(ROOT).as_posix())
    payload["metadata"]["materialized_asset_count"] = len(materialized_cache)
    return payload


def _report_selection_diagnostics(payload: Dict[str, Any]) -> Dict[str, Any]:
    case_selection = []
    for case in payload.get("verify_cases", []):
        entry = {
            "case_id": case.get("case_id"),
            "dataset": case.get("dataset"),
            "split": case.get("split"),
        }
        entry.update(dict(case.get("selection_diagnostics") or {}))
        case_selection.append(entry)

    dataset_verify_selection = []
    for dataset in payload.get("source_datasets", []):
        for summary in dataset.get("verify_selection_diagnostics", []) or []:
            dataset_verify_selection.append(dict(summary))

    return {
        "catalog_build_health": dict(payload.get("metadata", {}).get("catalog_build_health", {}) or {}),
        "case_selection": case_selection,
        "dataset_verify_selection": dataset_verify_selection,
    }


def validate_catalog_payload(payload: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    validator = Draft202012Validator(schema)
    schema_errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    for err in schema_errors:
        path = ".".join(str(p) for p in err.absolute_path) or "$"
        errors.append(f"Schema validation failed at {path}: {err.message}")

    verify_cases = payload.get("verify_cases", [])
    identify_gallery = payload.get("identify_gallery", {})
    identities = identify_gallery.get("identities", [])
    browser_seed = payload.get("dataset_browser_seed", [])
    case_ids = [item["case_id"] for item in verify_cases]
    identity_ids = [item["identity_id"] for item in identities]
    if len(case_ids) != len(set(case_ids)):
        errors.append("Duplicate case_id values detected.")
    if len(identity_ids) != len(set(identity_ids)):
        errors.append("Duplicate identity_id values detected.")

    required_case_types = {
        "easy_genuine",
        "hard_genuine",
        "hard_impostor",
        "cross_modality_genuine",
        "cross_modality_impostor",
    }
    present_case_types = {item["case_type"] for item in verify_cases}
    missing_case_types = sorted(required_case_types - present_case_types)
    if missing_case_types:
        errors.append(f"Missing required verify case categories: {missing_case_types}")

    included_datasets = set(payload.get("metadata", {}).get("included_datasets", []))
    expected_datasets = set(DATASET_META.keys())
    if expected_datasets - included_datasets:
        errors.append(f"Datasets missing from included_datasets metadata: {sorted(expected_datasets - included_datasets)}")

    asset_records: List[Tuple[str, Dict[str, Any]]] = []
    for case in verify_cases:
        if not case.get("source_pair_file"):
            errors.append(f"Verify case {case.get('case_id')} is missing source_pair_file.")
        asset_records.append((f"verify_cases.{case.get('case_id')}.image_a", case["image_a"]))
        asset_records.append((f"verify_cases.{case.get('case_id')}.image_b", case["image_b"]))
        if case.get("availability_status") != "available":
            errors.append(f"Verify case {case.get('case_id')} must be available but is {case.get('availability_status')}.")

    for identity in identities:
        for asset in identity.get("exemplars", []):
            asset_records.append((f"identify_gallery.identities.{identity.get('identity_id')}.{asset.get('asset_id')}", asset))

    for scenario in identify_gallery.get("demo_scenarios", []):
        asset_records.append((f"identify_gallery.demo_scenarios.{scenario.get('scenario_id')}.probe_asset", scenario["probe_asset"]))

    for entry in browser_seed:
        for asset in entry.get("items", []):
            asset_records.append((f"dataset_browser_seed.{entry.get('dataset')}.{asset.get('asset_id')}", asset))

    non_available_assets: List[str] = []
    missing_local_paths: List[str] = []
    missing_thumbnail_paths: List[str] = []
    for location, asset in asset_records:
        status = str(asset.get("availability_status"))
        if status != "available":
            non_available_assets.append(f"{location} -> {status}")
        rel_path = asset.get("relative_path") or asset.get("path")
        abs_path = (ROOT / rel_path).resolve() if rel_path else None
        if not rel_path or not abs_path or not abs_path.exists():
            missing_local_paths.append(f"{location} -> {rel_path}")
        thumbnail_path = asset.get("thumbnail_path")
        thumbnail_abs_path = (ROOT / thumbnail_path).resolve() if thumbnail_path else None
        if not thumbnail_path or not thumbnail_abs_path or not thumbnail_abs_path.exists():
            missing_thumbnail_paths.append(f"{location} -> {thumbnail_path}")
        detail = asset.get("availability_detail", {})
        if not detail.get("traceable_to_manifest"):
            errors.append(f"Asset {location} lost manifest traceability.")
        if not asset.get("source_path"):
            errors.append(f"Asset {location} is missing source_path provenance.")

    if non_available_assets:
        errors.append("Non-available curated assets detected: " + "; ".join(non_available_assets[:10]))
    if missing_local_paths:
        errors.append("Curated asset local paths missing on disk: " + "; ".join(missing_local_paths[:10]))
    if missing_thumbnail_paths:
        errors.append("Curated asset thumbnails missing on disk: " + "; ".join(missing_thumbnail_paths[:10]))

    scenario_types = {item["scenario_type"] for item in identify_gallery.get("demo_scenarios", [])}
    required_scenarios = {"positive_identification", "difficult_identification", "no_match"}
    if scenario_types != required_scenarios:
        errors.append(f"Identification scenario coverage mismatch. Expected {sorted(required_scenarios)}, got {sorted(scenario_types)}")

    if not verify_cases:
        errors.append("verify_cases must not be empty.")
    if not identities:
        errors.append("identify_gallery.identities must not be empty.")
    if not browser_seed:
        errors.append("dataset_browser_seed must not be empty.")

    selection_diagnostics = _report_selection_diagnostics(payload)
    for summary in selection_diagnostics.get("dataset_verify_selection", []):
        dataset_split = f"{summary.get('dataset')}:{summary.get('split')}"
        if summary.get("planned_curated_cases") and not summary.get("benchmark_best_available"):
            warnings.append(
                f"Curated verify selection for {dataset_split} had no benchmark-best evidence; heuristic degradation was used."
            )
        elif summary.get("planned_curated_cases") and not summary.get("benchmark_resolution_complete"):
            warnings.append(
                f"Curated verify selection for {dataset_split} had incomplete benchmark-best evidence."
            )
        if summary.get("heuristic_fallback_used"):
            warnings.append(f"Curated verify selection for {dataset_split} used heuristic fallback.")
        if summary.get("score_files_missing"):
            warnings.append(f"Curated verify selection for {dataset_split} was missing benchmark score files.")
        if summary.get("score_files_unparsable"):
            warnings.append(f"Curated verify selection for {dataset_split} had unparsable benchmark score files.")
        if summary.get("score_files_missing_columns"):
            warnings.append(f"Curated verify selection for {dataset_split} had benchmark score files missing required columns.")
        if summary.get("score_files_no_pair_overlap"):
            warnings.append(f"Curated verify selection for {dataset_split} had benchmark score files with no curated pair overlap.")
        planned = int(summary.get("planned_curated_cases", 0) or 0)
        built = int(summary.get("curated_cases_built_successfully", 0) or 0)
        if planned and built < planned:
            warnings.append(f"Curated verify selection for {dataset_split} built only {built} of {planned} planned cases.")
    warnings = list(dict.fromkeys(warnings))

    status = "pass" if not errors else "fail"
    return {
        "validation_status": status,
        "schema_errors_count": len(schema_errors),
        "validation_errors_count": len(errors),
        "validation_warnings_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "unique_case_ids": len(case_ids) == len(set(case_ids)),
            "unique_identity_ids": len(identity_ids) == len(set(identity_ids)),
            "verify_case_count": len(verify_cases),
            "identity_count": len(identities),
            "browser_seed_count": sum(len(entry.get("items", [])) for entry in browser_seed),
            "materialized_asset_count": payload.get("metadata", {}).get("materialized_asset_count", 0),
            "all_curated_assets_available": not non_available_assets,
            "all_curated_asset_paths_exist": not missing_local_paths,
            "all_curated_asset_thumbnails_exist": not missing_thumbnail_paths,
            "present_case_types": sorted(present_case_types),
            "present_scenario_types": sorted(scenario_types),
        },
        "selection_diagnostics": selection_diagnostics,
    }



def build_catalog_bundle(*, generated_at: Optional[str] = None, write_files: bool = True) -> Dict[str, Any]:
    benchmark_best = load_benchmark_best()
    bundles = {dataset: load_dataset_bundle(dataset, benchmark_best) for dataset in IDENTITY_DATASET_ORDER}

    verify_cases = [case for plan in VERIFY_CASE_PLANS if (case := build_verify_case(bundles[plan["dataset"]], plan)) is not None]
    identity_records = [build_identity_record(bundles[dataset]) for dataset in IDENTITY_DATASET_ORDER]
    scenarios = build_identification_scenarios(bundles, identity_records)
    identify_gallery = IdentifyGallery(identities=identity_records, demo_scenarios=scenarios)
    browser_seed = [build_browser_seed(bundles[dataset]) for dataset in IDENTITY_DATASET_ORDER]
    verify_selection_diagnostics_by_dataset, catalog_build_health = build_verify_selection_diagnostics(bundles, verify_cases)
    source_datasets = [
        build_source_dataset(
            bundles[dataset],
            verify_cases,
            identity_records,
            next(entry for entry in browser_seed if entry.dataset == dataset),
            verify_selection_diagnostics=verify_selection_diagnostics_by_dataset.get(dataset, []),
        )
        for dataset in IDENTITY_DATASET_ORDER
    ]

    generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    metadata = Metadata(
        catalog_version=CATALOG_VERSION,
        generation_policy_version=GENERATION_POLICY_VERSION,
        total_verify_cases=len(verify_cases),
        total_identity_records=len(identity_records),
        total_browser_seed_items=sum(len(entry.items) for entry in browser_seed),
        included_datasets=IDENTITY_DATASET_ORDER,
        excluded_datasets=[],
        validation_status="pending",
        validation_errors_count=0,
        validation_warnings_count=0,
        materialized_asset_root=str(ASSETS_ROOT.relative_to(ROOT).as_posix()),
        materialized_asset_count=0,
        limitations=[
            "The curated catalog ships deterministic local binary PNG assets plus thumbnails under data/samples/assets/.",
            "Each curated asset keeps upstream manifest traceability via source_path and traceability metadata.",
            "The catalog is deterministic for a fixed set of manifests and benchmark reports.",
            "The curated verify_cases set intentionally stays fixed at 7 public stories; other datasets remain catalog-included for identity and browser coverage even when they are not verify-demo-exposed.",
        ],
        catalog_build_health=catalog_build_health,
    )
    catalog = CatalogModel(
        catalog_version=CATALOG_VERSION,
        generated_at=generated_at,
        source_datasets=source_datasets,
        verify_cases=verify_cases,
        identify_gallery=identify_gallery,
        dataset_browser_seed=browser_seed,
        metadata=metadata,
    )
    schema = CatalogModel.model_json_schema()
    payload = catalog.model_dump(mode="json")
    payload = materialize_catalog_assets(payload)
    report = validate_catalog_payload(payload, schema)
    payload["metadata"]["validation_status"] = report["validation_status"]
    payload["metadata"]["validation_errors_count"] = report["validation_errors_count"]
    payload["metadata"]["validation_warnings_count"] = report["validation_warnings_count"]

    if write_files:
        _json_dump(SAMPLES_ROOT / "catalog.json", payload)
        _json_dump(SAMPLES_ROOT / "catalog.schema.json", schema)
        _json_dump(SAMPLES_ROOT / "catalog.validation_report.json", report)
        write_readme(payload, report)

    return {"catalog": payload, "schema": schema, "report": report}



def write_readme(payload: Dict[str, Any], report: Dict[str, Any]) -> None:
    lines = [
        "# Central Demo Catalog",
        "",
        "`data/samples/catalog.json` is the single source of truth for curated demo-ready fingerprint cases.",
        "",
        "## What this catalog contains",
        "",
        "- `verify_cases`: curated 1:1 verification stories with traceability back to `pairs_<split>.csv` and manifest rows.",
        "- `identify_gallery`: curated identities plus scenario seeds for positive, difficult, and no-match 1:N demos.",
        "- `dataset_browser_seed`: a small, deterministic, diverse subset per dataset for future browsing UX.",
        "- `source_datasets`: authoritative provenance describing which manifests, stats, split metadata and benchmark runs were used.",
        "- `data/samples/assets/`: deterministic local PNG assets plus thumbnails for every curated asset referenced by the catalog.",
        "- `verify_cases` intentionally stay narrow and curated; broader datasets can still appear under identity and browser coverage without being verify-demo-exposed.",
        "",
        "## Source of truth and relation to existing project data",
        "",
        "- The catalog is derived from `data/manifests/<dataset>/manifest.csv`, `pairs_*.csv`, `stats.json`, `split.json`, and `protocol_note.md` when present.",
        "- Where benchmark evidence exists, recommended methods and difficulty ordering are anchored to `artifacts/reports/benchmark/...`.",
        "- `catalog.json` is **not** a raw dump of manifests. It is a curated layer that references official project artifacts and adds deterministic demo semantics.",
        "- `processed/` and raw corpora remain upstream storage layers; the catalog preserves them via `source_path` / `traceability` while the consumer-facing `path` points to the local curated asset layer.",
        "- The local curated asset layer stores runnable binary PNG assets and thumbnails while preserving manifest-backed provenance.",
        "- Datasets outside the 7 curated verify stories remain catalog-included for identity and browser coverage when canonical assets exist.",
        "",
        "## Forward-compatibility contract",
        "",
        "Consumers may rely on:",
        "",
        "- stable top-level regions: `source_datasets`, `verify_cases`, `identify_gallery`, `dataset_browser_seed`, `metadata`",
        "- stable IDs: `case_id`, `identity_id`, `asset_id`, `scenario_id`",
        "- machine-verifiable structure from `catalog.schema.json`",
        "- explicit availability semantics via `availability_status` + `availability_detail`",
        "- `path` being a local, shipped artifact and `source_path` being the upstream provenance pointer",
        "",
        "Consumers must **not** assume:",
        "",
        "- that every dataset has the same finger semantics (`frgp=0` exists in some public sources)",
        "- that every local curated asset preserves the original pixel matrix byte-for-byte; some assets are deterministically re-rendered to PNG for UI compatibility",
        "- that `is_demo_safe=true` for real biometric datasets without an explicit usage review",
        "- that path layout should be re-derived manually from directory structure instead of reading the catalog",
        "",
        "## Validation summary",
        "",
        f"- Validation status: `{report['validation_status']}`",
        f"- Errors: `{report['validation_errors_count']}`",
        f"- Warnings: `{report['validation_warnings_count']}`",
        f"- Materialized curated assets: `{payload['metadata'].get('materialized_asset_count', 0)}`",
        "",
        "## Regeneration",
        "",
        "Run:",
        "",
        "```bash",
        "python scripts/build_demo_catalog.py",
        "```",
        "",
        "This rewrites `catalog.json`, `catalog.schema.json`, `catalog.validation_report.json`, and refreshes the deterministic local asset layer under `data/samples/assets/`.",
    ]
    readme_path = SAMPLES_ROOT / "README_catalog.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
