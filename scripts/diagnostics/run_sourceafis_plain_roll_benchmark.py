from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fpbench.fingerprint_engine import FingerprintImage, FingerprintTemplate, get_engine
from src.fpbench.fingerprint_engine.base import FingerprintEngine
from src.fpbench.fingerprint_engine.errors import FingerprintEngineError
from src.fpbench.fingerprint_engine.providers.sourceafis_provider import (
    SOURCEAFIS_ENABLED_ENV,
    SOURCEAFIS_SERVICE_URL_ENV,
)
from src.fpbench.fingerprint_engine.types import EngineMetadata


PROVIDER_ID = "sourceafis_open"
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_SPLITS = ("val", "test")
TARGET_FARS = (0.01, 0.005, 0.001, 0.0001)
OUTPUT_SCHEMA_VERSION = "sourceafis_open_plain_roll_benchmark_v1"
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "sourceafis_open_plain_roll_v1"
)

SCORES_COLUMNS = [
    "dataset",
    "split",
    "pair_id",
    "label",
    "is_positive",
    "subject_a",
    "subject_b",
    "finger_position",
    "path_a",
    "path_b",
    "raw_score",
    "score_semantics",
    "higher_is_more_similar",
    "provider_id",
    "provider_version",
    "template_format",
    "template_version",
    "extraction_cache_hit_a",
    "extraction_cache_hit_b",
    "extraction_latency_ms_a",
    "extraction_latency_ms_b",
    "verification_latency_ms",
    "verification_wall_latency_ms",
    "normalized_score_returned",
    "warnings",
    "error",
]

THRESHOLD_COLUMNS = [
    "dataset",
    "target_far",
    "threshold",
    "calibration_split",
    "calibration_negative_count",
    "calibration_positive_count",
    "calibration_false_accepts",
    "calibration_far",
    "enough_negatives_for_target",
    "minimum_negatives_for_target",
    "selection_rule",
    "higher_is_more_similar",
]

METRICS_COLUMNS = [
    "dataset",
    "split",
    "target_far",
    "threshold",
    "threshold_split",
    "threshold_val_far",
    "threshold_val_false_accepts",
    "tar",
    "far",
    "frr",
    "ta",
    "fr",
    "fa",
    "tr",
    "n_positive",
    "n_negative",
    "n_scored",
    "n_unscored",
    "auc",
    "eer",
    "eer_threshold",
    "enough_negatives_for_target",
    "minimum_negatives_for_target",
    "score_count",
    "score_min",
    "score_p05",
    "score_p25",
    "score_median",
    "score_mean",
    "score_p75",
    "score_p95",
    "score_max",
    "positive_score_mean",
    "negative_score_mean",
]

LATENCY_COLUMNS = [
    "dataset",
    "split",
    "operation",
    "status",
    "count",
    "cache_hits",
    "cache_misses",
    "min_ms",
    "p50_ms",
    "mean_ms",
    "p95_ms",
    "max_ms",
    "total_ms",
]

FAILURE_COLUMNS = [
    "dataset",
    "split",
    "pair_id",
    "operation",
    "path",
    "subject_a",
    "subject_b",
    "finger_position",
    "error_type",
    "error_message",
]


class SourceAfisBenchmarkError(RuntimeError):
    """Raised for benchmark setup or protocol failures."""


@dataclass(frozen=True)
class ThresholdSelection:
    target_far: float
    threshold: float
    false_accepts: int
    actual_far: float
    negative_count: int
    positive_count: int
    enough_negatives_for_target: bool
    minimum_negatives_for_target: int


@dataclass(frozen=True)
class TemplateCacheResult:
    template: FingerprintTemplate
    image_path: str
    image_sha256: str
    cache_key: str
    cache_hit: bool
    latency_ms: float


def output_schema() -> dict[str, list[str]]:
    return {
        "sourceafis_plain_roll_scores_val.csv": list(SCORES_COLUMNS),
        "sourceafis_plain_roll_scores_test.csv": list(SCORES_COLUMNS),
        "sourceafis_plain_roll_thresholds.csv": list(THRESHOLD_COLUMNS),
        "sourceafis_plain_roll_metrics.csv": list(METRICS_COLUMNS),
        "sourceafis_plain_roll_latency_summary.csv": list(LATENCY_COLUMNS),
        "sourceafis_plain_roll_failures.csv": list(FAILURE_COLUMNS),
    }


def parse_file_uri(raw: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _finite_labels_scores(labels: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores_arr) & np.isin(labels_arr, [0, 1])
    return labels_arr[valid], scores_arr[valid]


def _minimum_negatives_for_target(target_far: float) -> int:
    if target_far <= 0:
        return math.inf  # type: ignore[return-value]
    return int(math.ceil(1.0 / float(target_far)))


def select_threshold_for_far(labels: Any, scores: Any, target_far: float) -> ThresholdSelection:
    """Select the most permissive VAL threshold whose empirical FAR is <= target.

    SourceAFIS scores are raw similarity scores, so higher values are treated as
    stronger matches. Ties are handled as a group because acceptance is
    score >= threshold.
    """

    target = float(target_far)
    if target < 0:
        raise ValueError("target_far must be non-negative.")

    labels_arr, scores_arr = _finite_labels_scores(labels, scores)
    positives = labels_arr == 1
    negatives = labels_arr == 0
    negative_scores = scores_arr[negatives]
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negatives))
    minimum_negatives = _minimum_negatives_for_target(target)
    enough_negatives = bool(negative_count >= minimum_negatives) if math.isfinite(minimum_negatives) else False

    if negative_count == 0 or scores_arr.size == 0:
        return ThresholdSelection(
            target_far=target,
            threshold=float("nan"),
            false_accepts=0,
            actual_far=float("nan"),
            negative_count=negative_count,
            positive_count=positive_count,
            enough_negatives_for_target=enough_negatives,
            minimum_negatives_for_target=int(minimum_negatives) if math.isfinite(minimum_negatives) else -1,
        )

    for threshold in sorted(float(x) for x in np.unique(scores_arr)):
        false_accepts = int(np.sum(negative_scores >= threshold))
        actual_far = false_accepts / negative_count
        if actual_far <= target + 1e-15:
            return ThresholdSelection(
                target_far=target,
                threshold=float(threshold),
                false_accepts=false_accepts,
                actual_far=float(actual_far),
                negative_count=negative_count,
                positive_count=positive_count,
                enough_negatives_for_target=enough_negatives,
                minimum_negatives_for_target=int(minimum_negatives) if math.isfinite(minimum_negatives) else -1,
            )

    threshold = math.nextafter(float(np.max(scores_arr)), math.inf)
    return ThresholdSelection(
        target_far=target,
        threshold=float(threshold),
        false_accepts=0,
        actual_far=0.0,
        negative_count=negative_count,
        positive_count=positive_count,
        enough_negatives_for_target=enough_negatives,
        minimum_negatives_for_target=int(minimum_negatives) if math.isfinite(minimum_negatives) else -1,
    )


def compute_confusion(labels: Any, scores: Any, threshold: float) -> dict[str, Any]:
    labels_arr, scores_arr = _finite_labels_scores(labels, scores)
    positives = labels_arr == 1
    negatives = labels_arr == 0
    if math.isfinite(float(threshold)):
        accepted = scores_arr >= float(threshold)
    else:
        accepted = np.zeros_like(scores_arr, dtype=bool)

    ta = int(np.sum(accepted & positives))
    fr = int(np.sum((~accepted) & positives))
    fa = int(np.sum(accepted & negatives))
    tr = int(np.sum((~accepted) & negatives))
    n_positive = int(np.sum(positives))
    n_negative = int(np.sum(negatives))
    tar = float(ta / n_positive) if n_positive else float("nan")
    far = float(fa / n_negative) if n_negative else float("nan")
    return {
        "tar": tar,
        "far": far,
        "frr": float(1.0 - tar) if math.isfinite(tar) else float("nan"),
        "ta": ta,
        "fr": fr,
        "fa": fa,
        "tr": tr,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def compute_auc(labels: Any, scores: Any) -> float:
    labels_arr, scores_arr = _finite_labels_scores(labels, scores)
    n_positive = int(np.sum(labels_arr == 1))
    n_negative = int(np.sum(labels_arr == 0))
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    ranks = np.empty(scores_arr.size, dtype=float)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(np.sum(ranks[labels_arr == 1]))
    auc = (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)
    return float(auc)


def compute_eer(labels: Any, scores: Any) -> tuple[float, float]:
    labels_arr, scores_arr = _finite_labels_scores(labels, scores)
    n_positive = int(np.sum(labels_arr == 1))
    n_negative = int(np.sum(labels_arr == 0))
    if n_positive == 0 or n_negative == 0 or scores_arr.size == 0:
        return float("nan"), float("nan")

    unique_scores = sorted((float(x) for x in np.unique(scores_arr)), reverse=True)
    thresholds = [math.nextafter(float(max(unique_scores)), math.inf), *unique_scores]
    points: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        counts = compute_confusion(labels_arr, scores_arr, threshold)
        far = float(counts["far"])
        frr = float(counts["frr"])
        points.append((far, frr, float(threshold)))

    previous = points[0]
    if math.isclose(previous[0], previous[1], rel_tol=0.0, abs_tol=1e-15):
        return float(previous[0]), float(previous[2])

    for current in points[1:]:
        prev_diff = previous[0] - previous[1]
        curr_diff = current[0] - current[1]
        if math.isclose(curr_diff, 0.0, rel_tol=0.0, abs_tol=1e-15):
            return float(current[0]), float(current[2])
        if (prev_diff < 0 <= curr_diff) or (prev_diff > 0 >= curr_diff):
            denom = curr_diff - prev_diff
            fraction = 0.0 if math.isclose(denom, 0.0) else -prev_diff / denom
            eer = previous[0] + fraction * (current[0] - previous[0])
            threshold = current[2]
            if math.isfinite(previous[2]) and math.isfinite(current[2]):
                threshold = previous[2] + fraction * (current[2] - previous[2])
            return float(eer), float(threshold)
        previous = current

    far, frr, threshold = min(points, key=lambda point: (abs(point[0] - point[1]), point[2]))
    return float((far + frr) / 2.0), float(threshold)


def compute_auc_eer(labels: Any, scores: Any) -> tuple[float, float, float]:
    auc = compute_auc(labels, scores)
    eer, eer_threshold = compute_eer(labels, scores)
    return auc, eer, eer_threshold


def score_distribution_summary(scores: Any) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "score_count": 0,
            "score_min": float("nan"),
            "score_p05": float("nan"),
            "score_p25": float("nan"),
            "score_median": float("nan"),
            "score_mean": float("nan"),
            "score_p75": float("nan"),
            "score_p95": float("nan"),
            "score_max": float("nan"),
        }
    return {
        "score_count": int(values.size),
        "score_min": float(np.min(values)),
        "score_p05": float(np.quantile(values, 0.05)),
        "score_p25": float(np.quantile(values, 0.25)),
        "score_median": float(np.median(values)),
        "score_mean": float(np.mean(values)),
        "score_p75": float(np.quantile(values, 0.75)),
        "score_p95": float(np.quantile(values, 0.95)),
        "score_max": float(np.max(values)),
    }


def ensure_sourceafis_available(
    engine: FingerprintEngine | None = None,
    *,
    require_enabled_env: bool = True,
) -> tuple[FingerprintEngine, EngineMetadata]:
    if require_enabled_env and not _truthy(os.getenv(SOURCEAFIS_ENABLED_ENV)):
        raise SourceAfisBenchmarkError(
            f"{SOURCEAFIS_ENABLED_ENV}=true is required for this benchmark. "
            "Start the SourceAFIS sidecar, set the environment variables, then rerun."
        )

    service_url = str(os.getenv(SOURCEAFIS_SERVICE_URL_ENV) or "").strip()
    if require_enabled_env and not service_url:
        raise SourceAfisBenchmarkError(
            f"{SOURCEAFIS_SERVICE_URL_ENV} must point to a reachable SourceAFIS HTTP sidecar."
        )

    selected_engine = engine or get_engine(PROVIDER_ID)
    metadata = selected_engine.metadata()
    if metadata.provider_id != PROVIDER_ID:
        raise SourceAfisBenchmarkError(
            f"Expected fingerprint engine provider {PROVIDER_ID!r}, got {metadata.provider_id!r}."
        )
    if not metadata.available:
        reason = metadata.unavailable_reason or "provider reported unavailable"
        raise SourceAfisBenchmarkError(f"{PROVIDER_ID} is unavailable: {reason}")
    return selected_engine, metadata


def validate_output_directory(outdir: Path, *, repo_root: Path = REPO_ROOT) -> Path:
    output = parse_file_uri(outdir, repo_root=repo_root)
    forbidden = (repo_root / "artifacts" / "reports" / "benchmark" / "current").resolve()
    try:
        output.relative_to(forbidden)
    except ValueError:
        return output
    raise SourceAfisBenchmarkError(
        "Refusing to write SourceAFIS benchmark outputs under artifacts/reports/benchmark/current."
    )


def _pairs_path(dataset: str, split: str, *, repo_root: Path = REPO_ROOT) -> Path | None:
    candidates = [
        repo_root / "data" / "manifests" / dataset / f"pairs_{split}.csv",
        repo_root / "data" / "manifests" / dataset / "pairs" / f"pairs_{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _one_plain_one_roll(path_a: Any, path_b: Any) -> bool:
    left = str(path_a).lower().replace("\\", "/")
    right = str(path_b).lower().replace("\\", "/")
    left_plain = "plain" in left
    right_plain = "plain" in right
    left_roll = "roll" in left or "rolled" in left
    right_roll = "roll" in right or "rolled" in right
    return (left_plain and right_roll) or (left_roll and right_plain)


def _limit_pairs(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    positives = df[df["label"] == 1].head(math.ceil(limit / 2))
    negatives = df[df["label"] == 0].head(math.floor(limit / 2))
    limited = pd.concat([positives, negatives], ignore_index=False).sort_values("_source_order")
    return limited.head(limit).reset_index(drop=True)


def load_plain_roll_pairs(
    dataset: str,
    split: str,
    *,
    repo_root: Path = REPO_ROOT,
    limit: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs_csv = _pairs_path(dataset, split, repo_root=repo_root)
    if pairs_csv is None:
        return pd.DataFrame(), {
            "dataset": dataset,
            "split": split,
            "compatible": False,
            "reason": "no pairs CSV found",
        }

    df = pd.read_csv(pairs_csv)
    required = {"pair_id", "label", "split", "subject_a", "subject_b", "path_a", "path_b"}
    missing = sorted(required - set(df.columns))
    if missing:
        return pd.DataFrame(), {
            "dataset": dataset,
            "split": split,
            "pairs_csv": str(pairs_csv),
            "compatible": False,
            "reason": f"missing required columns: {missing}",
            "n_pairs": int(len(df)),
        }

    finger_col = "frgp" if "frgp" in df.columns else "finger_id" if "finger_id" in df.columns else None
    if finger_col is None:
        return pd.DataFrame(), {
            "dataset": dataset,
            "split": split,
            "pairs_csv": str(pairs_csv),
            "compatible": False,
            "reason": "missing frgp/finger_id column for same-finger protocol",
            "n_pairs": int(len(df)),
        }

    normalized = df.copy()
    normalized["_source_order"] = np.arange(len(normalized))
    normalized["label"] = pd.to_numeric(normalized["label"], errors="coerce").fillna(-1).astype(int)
    normalized["split"] = normalized["split"].astype(str).str.strip().str.lower()
    normalized["subject_a"] = normalized["subject_a"].astype(str)
    normalized["subject_b"] = normalized["subject_b"].astype(str)
    normalized["finger_position"] = normalized[finger_col].astype(str)
    normalized["dataset"] = dataset

    split_mask = normalized["split"] == split.lower()
    label_mask = normalized["label"].isin([0, 1])
    plain_roll_mask = normalized.apply(lambda row: _one_plain_one_roll(row["path_a"], row["path_b"]), axis=1)
    same_subject = normalized["subject_a"] == normalized["subject_b"]
    protocol_mask = ((normalized["label"] == 1) & same_subject) | ((normalized["label"] == 0) & ~same_subject)
    filtered = normalized[split_mask & label_mask & plain_roll_mask & protocol_mask].copy()
    filtered = _limit_pairs(filtered, int(limit))

    status = {
        "dataset": dataset,
        "split": split,
        "pairs_csv": str(pairs_csv),
        "compatible": bool(len(filtered)),
        "reason": "positive/negative same-finger plain-vs-roll pairs"
        if len(filtered)
        else "no pairs survived plain-vs-roll protocol filtering",
        "n_pairs": int(len(filtered)),
        "n_positive": int((filtered["label"] == 1).sum()) if len(filtered) else 0,
        "n_negative": int((filtered["label"] == 0).sum()) if len(filtered) else 0,
        "source_n_pairs": int(len(df)),
        "filtered_out_pairs": int(len(df) - len(filtered)),
    }
    columns = [
        "dataset",
        "split",
        "pair_id",
        "label",
        "subject_a",
        "subject_b",
        "finger_position",
        "path_a",
        "path_b",
    ]
    return filtered[columns].reset_index(drop=True), status


def _image_mime_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix)


def _capture_type(path: Path) -> str | None:
    value = str(path).lower().replace("\\", "/")
    if "plain" in value:
        return "plain"
    if "roll" in value or "rolled" in value:
        return "roll"
    return None


def _read_image(path: Path) -> tuple[bytes, str]:
    image_bytes = path.read_bytes()
    return image_bytes, hashlib.sha256(image_bytes).hexdigest()


def _metadata_value(metadata: EngineMetadata, key: str, default: str = "") -> str:
    value = metadata.metadata.get(key)
    if value in (None, ""):
        return default
    return str(value)


class TemplateCache:
    def __init__(
        self,
        cache_dir: Path,
        *,
        provider_metadata: EngineMetadata,
        service_url: str,
        repo_root: Path = REPO_ROOT,
    ) -> None:
        self.cache_dir = cache_dir
        self.provider_metadata = provider_metadata
        self.service_url = service_url
        self.repo_root = repo_root
        self._memory: dict[str, TemplateCacheResult] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        *,
        engine: FingerprintEngine,
        image_path: str | Path,
        dataset: str,
        split: str,
        pair_id: Any,
        side: str,
    ) -> TemplateCacheResult:
        resolved_path = parse_file_uri(image_path, repo_root=self.repo_root)
        image_bytes, image_sha256 = _read_image(resolved_path)
        cache_key = self._cache_key(resolved_path, image_sha256)
        if cache_key in self._memory:
            cached = self._memory[cache_key]
            return TemplateCacheResult(
                template=cached.template,
                image_path=cached.image_path,
                image_sha256=cached.image_sha256,
                cache_key=cached.cache_key,
                cache_hit=True,
                latency_ms=0.0,
            )

        cache_path = self.cache_dir / f"{cache_key}.json"
        start = time.perf_counter()
        if cache_path.exists():
            template = self._load_cache_file(cache_path, image_sha256=image_sha256)
            latency_ms = (time.perf_counter() - start) * 1000.0
            result = TemplateCacheResult(
                template=template,
                image_path=str(resolved_path),
                image_sha256=image_sha256,
                cache_key=cache_key,
                cache_hit=True,
                latency_ms=float(latency_ms),
            )
            self._memory[cache_key] = result
            return result

        image = FingerprintImage(
            image_bytes=image_bytes,
            sha256=image_sha256,
            path=str(resolved_path),
            image_id=f"{dataset}:{split}:{pair_id}:{side}:{image_sha256[:16]}",
            mime_type=_image_mime_type(resolved_path),
            capture_type=_capture_type(resolved_path),
            metadata={
                "dataset": dataset,
                "split": split,
                "pair_id": str(pair_id),
                "side": side,
                "path": str(resolved_path),
            },
        )
        start = time.perf_counter()
        template = engine.extract_template(image)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._write_cache_file(cache_path, template, resolved_path, image_sha256)
        result = TemplateCacheResult(
            template=template,
            image_path=str(resolved_path),
            image_sha256=image_sha256,
            cache_key=cache_key,
            cache_hit=False,
            latency_ms=float(latency_ms),
        )
        self._memory[cache_key] = result
        return result

    def _cache_key(self, image_path: Path, image_sha256: str) -> str:
        payload = {
            "provider_id": PROVIDER_ID,
            "provider_version": self.provider_metadata.provider_version,
            "template_format": self.provider_metadata.template_format,
            "template_version": self.provider_metadata.template_version,
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            "service_url": self.service_url,
            "engine_version": self.provider_metadata.provider_version,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_cache_file(self, cache_path: Path, *, image_sha256: str) -> FingerprintTemplate:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("provider_id") != PROVIDER_ID:
            raise SourceAfisBenchmarkError(f"Template cache provider mismatch in {cache_path}.")
        if payload.get("image_sha256") != image_sha256:
            raise SourceAfisBenchmarkError(f"Template cache image hash mismatch in {cache_path}.")
        template_bytes = base64.b64decode(str(payload["template_base64"]).encode("ascii"), validate=True)
        return FingerprintTemplate(
            provider_id=PROVIDER_ID,
            provider_version=str(payload.get("provider_version") or self.provider_metadata.provider_version),
            template_format=str(payload.get("template_format") or "sourceafis"),
            template_version=str(payload.get("template_version") or self.provider_metadata.template_version or "unknown"),
            template_bytes=template_bytes,
            image_id=str(payload.get("image_id") or ""),
            quality_score=None,
            metadata={
                "runtime": "sourceafis_http_sidecar",
                "loaded_from_template_cache": True,
                "cache_key": cache_path.stem,
            },
        )

    def _write_cache_file(
        self,
        cache_path: Path,
        template: FingerprintTemplate,
        image_path: Path,
        image_sha256: str,
    ) -> None:
        payload = {
            "schema_version": "sourceafis_template_cache_v1",
            "provider_id": template.provider_id,
            "provider_version": template.provider_version,
            "template_format": template.template_format,
            "template_version": template.template_version,
            "image_id": template.image_id,
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            "service_url": self.service_url,
            "engine_version": self.provider_metadata.provider_version,
            "created_at": _utc_now(),
            "template_size_bytes": len(template.template_bytes),
            "template_base64": base64.b64encode(template.template_bytes).decode("ascii"),
        }
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _latency_event(
    dataset: str,
    split: str,
    operation: str,
    status: str,
    latency_ms: float,
    *,
    cache_hit: bool | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "operation": operation,
        "status": status,
        "latency_ms": float(latency_ms),
        "cache_hit": cache_hit,
    }


def score_pairs(
    pairs: pd.DataFrame,
    *,
    engine: FingerprintEngine,
    cache: TemplateCache,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    latency_events: list[dict[str, Any]] = []

    for pair in pairs.itertuples(index=False):
        dataset = str(pair.dataset)
        split = str(pair.split)
        pair_id = str(pair.pair_id)
        row_base = {
            "dataset": dataset,
            "split": split,
            "pair_id": pair_id,
            "label": int(pair.label),
            "is_positive": bool(int(pair.label) == 1),
            "subject_a": str(pair.subject_a),
            "subject_b": str(pair.subject_b),
            "finger_position": str(pair.finger_position),
            "path_a": str(pair.path_a),
            "path_b": str(pair.path_b),
            "score_semantics": "sourceafis_raw_similarity_score",
            "higher_is_more_similar": True,
            "provider_id": PROVIDER_ID,
            "provider_version": "",
            "template_format": "sourceafis",
            "template_version": "",
            "warnings": "",
        }
        try:
            template_a = cache.get(
                engine=engine,
                image_path=pair.path_a,
                dataset=dataset,
                split=split,
                pair_id=pair_id,
                side="a",
            )
            latency_events.append(
                _latency_event(
                    dataset,
                    split,
                    "template_cache_lookup" if template_a.cache_hit else "template_extraction",
                    "ok",
                    template_a.latency_ms,
                    cache_hit=template_a.cache_hit,
                )
            )
            template_b = cache.get(
                engine=engine,
                image_path=pair.path_b,
                dataset=dataset,
                split=split,
                pair_id=pair_id,
                side="b",
            )
            latency_events.append(
                _latency_event(
                    dataset,
                    split,
                    "template_cache_lookup" if template_b.cache_hit else "template_extraction",
                    "ok",
                    template_b.latency_ms,
                    cache_hit=template_b.cache_hit,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "pair_id": pair_id,
                    "operation": "extract_template",
                    "path": f"{pair.path_a} | {pair.path_b}",
                    "subject_a": str(pair.subject_a),
                    "subject_b": str(pair.subject_b),
                    "finger_position": str(pair.finger_position),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            rows.append(
                {
                    **row_base,
                    "raw_score": float("nan"),
                    "extraction_cache_hit_a": "",
                    "extraction_cache_hit_b": "",
                    "extraction_latency_ms_a": float("nan"),
                    "extraction_latency_ms_b": float("nan"),
                    "verification_latency_ms": float("nan"),
                    "verification_wall_latency_ms": float("nan"),
                    "normalized_score_returned": False,
                    "error": f"template extraction failed: {exc}",
                }
            )
            latency_events.append(_latency_event(dataset, split, "template_extraction", "failed", 0.0))
            continue

        try:
            start = time.perf_counter()
            match = engine.verify(template_a.template, template_b.template)
            wall_latency_ms = (time.perf_counter() - start) * 1000.0
            verification_latency_ms = (
                float(match.latency_ms) if match.latency_ms is not None else float(wall_latency_ms)
            )
            latency_events.append(
                _latency_event(dataset, split, "verification", "ok", verification_latency_ms)
            )
            rows.append(
                {
                    **row_base,
                    "raw_score": float(match.score),
                    "provider_version": match.provider_version,
                    "template_format": template_a.template.template_format,
                    "template_version": template_a.template.template_version,
                    "extraction_cache_hit_a": bool(template_a.cache_hit),
                    "extraction_cache_hit_b": bool(template_b.cache_hit),
                    "extraction_latency_ms_a": float(template_a.latency_ms),
                    "extraction_latency_ms_b": float(template_b.latency_ms),
                    "verification_latency_ms": verification_latency_ms,
                    "verification_wall_latency_ms": float(wall_latency_ms),
                    "normalized_score_returned": match.normalized_score is not None,
                    "warnings": "; ".join(match.warnings),
                    "error": "",
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "pair_id": pair_id,
                    "operation": "verify",
                    "path": "",
                    "subject_a": str(pair.subject_a),
                    "subject_b": str(pair.subject_b),
                    "finger_position": str(pair.finger_position),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            latency_events.append(_latency_event(dataset, split, "verification", "failed", 0.0))
            rows.append(
                {
                    **row_base,
                    "raw_score": float("nan"),
                    "template_format": template_a.template.template_format,
                    "template_version": template_a.template.template_version,
                    "extraction_cache_hit_a": bool(template_a.cache_hit),
                    "extraction_cache_hit_b": bool(template_b.cache_hit),
                    "extraction_latency_ms_a": float(template_a.latency_ms),
                    "extraction_latency_ms_b": float(template_b.latency_ms),
                    "verification_latency_ms": float("nan"),
                    "verification_wall_latency_ms": float("nan"),
                    "normalized_score_returned": False,
                    "error": f"verification failed: {exc}",
                }
            )

    return pd.DataFrame(rows), failures, latency_events


def build_threshold_table(scores: pd.DataFrame, target_fars: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scores.empty:
        return pd.DataFrame(columns=THRESHOLD_COLUMNS)

    for dataset, dataset_scores in scores.groupby("dataset", sort=True):
        val = dataset_scores[dataset_scores["split"].astype(str).str.lower() == "val"].copy()
        labels = pd.to_numeric(val["label"], errors="coerce").fillna(-1).to_numpy(dtype=int)
        raw_scores = pd.to_numeric(val["raw_score"], errors="coerce").to_numpy(dtype=float)
        for target_far in target_fars:
            selection = select_threshold_for_far(labels, raw_scores, float(target_far))
            rows.append(
                {
                    "dataset": dataset,
                    "target_far": float(target_far),
                    "threshold": float(selection.threshold),
                    "calibration_split": "val",
                    "calibration_negative_count": int(selection.negative_count),
                    "calibration_positive_count": int(selection.positive_count),
                    "calibration_false_accepts": int(selection.false_accepts),
                    "calibration_far": float(selection.actual_far),
                    "enough_negatives_for_target": bool(selection.enough_negatives_for_target),
                    "minimum_negatives_for_target": int(selection.minimum_negatives_for_target),
                    "selection_rule": "lowest VAL raw-score threshold with VAL FAR <= target",
                    "higher_is_more_similar": True,
                }
            )
    return pd.DataFrame(rows, columns=THRESHOLD_COLUMNS)


def build_metrics_table(scores: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scores.empty or thresholds.empty:
        return pd.DataFrame(columns=METRICS_COLUMNS)

    for threshold_row in thresholds.itertuples(index=False):
        dataset_scores = scores[scores["dataset"] == threshold_row.dataset].copy()
        for split in DEFAULT_SPLITS:
            split_scores = dataset_scores[dataset_scores["split"].astype(str).str.lower() == split].copy()
            labels_all = pd.to_numeric(split_scores["label"], errors="coerce").fillna(-1).to_numpy(dtype=int)
            raw_scores_all = pd.to_numeric(split_scores["raw_score"], errors="coerce").to_numpy(dtype=float)
            labels, raw_scores = _finite_labels_scores(labels_all, raw_scores_all)
            counts = compute_confusion(labels, raw_scores, float(threshold_row.threshold))
            auc, eer, eer_threshold = compute_auc_eer(labels, raw_scores)
            summary = score_distribution_summary(raw_scores)
            positive_scores = raw_scores[labels == 1]
            negative_scores = raw_scores[labels == 0]
            rows.append(
                {
                    "dataset": threshold_row.dataset,
                    "split": split,
                    "target_far": float(threshold_row.target_far),
                    "threshold": float(threshold_row.threshold),
                    "threshold_split": "val",
                    "threshold_val_far": float(threshold_row.calibration_far),
                    "threshold_val_false_accepts": int(threshold_row.calibration_false_accepts),
                    "tar": float(counts["tar"]),
                    "far": float(counts["far"]),
                    "frr": float(counts["frr"]),
                    "ta": int(counts["ta"]),
                    "fr": int(counts["fr"]),
                    "fa": int(counts["fa"]),
                    "tr": int(counts["tr"]),
                    "n_positive": int(counts["n_positive"]),
                    "n_negative": int(counts["n_negative"]),
                    "n_scored": int(raw_scores.size),
                    "n_unscored": int(len(split_scores) - raw_scores.size),
                    "auc": float(auc),
                    "eer": float(eer),
                    "eer_threshold": float(eer_threshold),
                    "enough_negatives_for_target": bool(threshold_row.enough_negatives_for_target),
                    "minimum_negatives_for_target": int(threshold_row.minimum_negatives_for_target),
                    **summary,
                    "positive_score_mean": float(np.mean(positive_scores)) if positive_scores.size else float("nan"),
                    "negative_score_mean": float(np.mean(negative_scores)) if negative_scores.size else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=METRICS_COLUMNS)


def build_latency_summary(latency_events: list[dict[str, Any]]) -> pd.DataFrame:
    if not latency_events:
        return pd.DataFrame(columns=LATENCY_COLUMNS)

    df = pd.DataFrame(latency_events)
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(["dataset", "split", "operation", "status"], sort=True):
        dataset, split, operation, status = keys
        values = pd.to_numeric(group["latency_ms"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        cache_hits = int((group["cache_hit"] == True).sum()) if "cache_hit" in group else 0
        cache_misses = int((group["cache_hit"] == False).sum()) if "cache_hit" in group else 0
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "operation": operation,
                "status": status,
                "count": int(len(group)),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "min_ms": float(np.min(values)) if values.size else float("nan"),
                "p50_ms": float(np.median(values)) if values.size else float("nan"),
                "mean_ms": float(np.mean(values)) if values.size else float("nan"),
                "p95_ms": float(np.quantile(values, 0.95)) if values.size else float("nan"),
                "max_ms": float(np.max(values)) if values.size else float("nan"),
                "total_ms": float(np.sum(values)) if values.size else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=LATENCY_COLUMNS)


def _write_csv(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=columns).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _fmt_float(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    return "nan" if not math.isfinite(number) else f"{number:.{digits}f}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    number = _safe_float(value)
    return "nan" if not math.isfinite(number) else f"{100.0 * number:.{digits}f}%"


def _metric_lookup(metrics: pd.DataFrame, dataset: str, split: str, target_far: float) -> pd.Series | None:
    rows = metrics[
        (metrics["dataset"] == dataset)
        & (metrics["split"] == split)
        & np.isclose(metrics["target_far"].astype(float), float(target_far))
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def render_summary_markdown(
    *,
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    latency: pd.DataFrame,
    failures: pd.DataFrame,
    dataset_statuses: list[dict[str, Any]],
    total_runtime_s: float,
    output_paths: dict[str, Path],
    comparison_created: bool,
) -> str:
    extraction_failures = int((failures["operation"] == "extract_template").sum()) if not failures.empty else 0
    scoring_failures = int((failures["operation"] == "verify").sum()) if not failures.empty else 0
    lines = [
        "# SourceAFIS Plain/Roll Benchmark",
        "",
        "Provider: `sourceafis_open` through the fingerprint_engine abstraction and the SourceAFIS HTTP sidecar.",
        "",
        "Score semantics: SourceAFIS raw similarity score; higher means a stronger match. Scores are not normalized. Thresholds are calibrated on VAL only and then applied unchanged to TEST.",
        "",
        f"Total runtime: {_fmt_float(total_runtime_s, 2)} s",
        f"Template extraction failures: {extraction_failures}",
        f"Scoring failures: {scoring_failures}",
        "",
        "## Dataset Protocols",
        "",
        "| dataset | split | compatible | pairs | positives | negatives | reason | pairs CSV |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for status in dataset_statuses:
        lines.append(
            f"| {status.get('dataset', '')} | {status.get('split', '')} | {bool(status.get('compatible', False))} | "
            f"{status.get('n_pairs', 0)} | {status.get('n_positive', 0)} | {status.get('n_negative', 0)} | "
            f"{status.get('reason', '')} | `{status.get('pairs_csv', '')}` |"
        )

    lines.extend(
        [
            "",
            "## TEST Operating Points",
            "",
            "| dataset | target FAR | threshold | VAL FAR | TEST TAR | TEST FAR | TEST FRR | TA | FR | FA | TR | negatives enough |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in metrics[metrics["split"] == "test"].sort_values(["dataset", "target_far"]).iterrows():
        enough = bool(row["enough_negatives_for_target"])
        lines.append(
            f"| {row['dataset']} | {_fmt_pct(row['target_far'])} | {_fmt_float(row['threshold'], 6)} | "
            f"{_fmt_pct(row['threshold_val_far'])} | {_fmt_pct(row['tar'])} | {_fmt_pct(row['far'])} | "
            f"{_fmt_pct(row['frr'])} | {int(row['ta'])} | {int(row['fr'])} | {int(row['fa'])} | "
            f"{int(row['tr'])} | {enough} |"
        )

    lines.extend(
        [
            "",
            "## AUC And EER",
            "",
            "| dataset | split | AUC | EER | EER threshold | scored pairs | unscored pairs |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    seen: set[tuple[str, str]] = set()
    for _, row in metrics.sort_values(["dataset", "split", "target_far"]).iterrows():
        key = (str(row["dataset"]), str(row["split"]))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {row['dataset']} | {row['split']} | {_fmt_float(row['auc'], 4)} | "
            f"{_fmt_pct(row['eer'])} | {_fmt_float(row['eer_threshold'], 6)} | "
            f"{int(row['n_scored'])} | {int(row['n_unscored'])} |"
        )

    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "| dataset | target FAR | threshold | VAL calibration FAR | false accepts / negatives | minimum negatives for target | enough negatives |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in thresholds.sort_values(["dataset", "target_far"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {_fmt_pct(row['target_far'])} | {_fmt_float(row['threshold'], 6)} | "
            f"{_fmt_pct(row['calibration_far'])} | {int(row['calibration_false_accepts'])}/{int(row['calibration_negative_count'])} | "
            f"{int(row['minimum_negatives_for_target'])} | {bool(row['enough_negatives_for_target'])} |"
        )

    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| dataset | split | operation | status | count | p50 ms | p95 ms | mean ms |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in latency.sort_values(["dataset", "split", "operation", "status"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['operation']} | {row['status']} | "
            f"{int(row['count'])} | {_fmt_float(row['p50_ms'], 3)} | {_fmt_float(row['p95_ms'], 3)} | "
            f"{_fmt_float(row['mean_ms'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Output Schema",
            "",
        ]
    )
    for filename, columns in output_schema().items():
        lines.append(f"- `{filename}`: {', '.join(columns)}")
    if comparison_created:
        lines.append("- `sourceafis_vs_sift_v2_comparison.csv`: optional comparison against existing SIFT v2 evidence.")
        lines.append("- `sourceafis_vs_sift_v2_comparison.md`: optional comparison against existing SIFT v2 evidence.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, path in output_paths.items():
        lines.append(f"- {key}: `{path}`")
    return "\n".join(lines) + "\n"


def _git_info(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> tuple[int, str]:
        proc = subprocess.run(["git", *args], cwd=str(repo_root), capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip()

    commit_rc, commit = run_git(["rev-parse", "HEAD"])
    branch_rc, branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status_rc, status = run_git(["status", "--porcelain"])
    return {
        "commit": commit if commit_rc == 0 else "",
        "branch": branch if branch_rc == 0 else "",
        "is_dirty": bool(status.strip()) if status_rc == 0 else None,
        "status_porcelain": status,
    }


def write_manifest(
    path: Path,
    *,
    provider_metadata: EngineMetadata,
    dataset_statuses: list[dict[str, Any]],
    output_paths: dict[str, Path],
    target_fars: tuple[float, ...],
    total_runtime_s: float,
    repo_root: Path,
    cache_dir: Path,
) -> None:
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "repo_root": str(repo_root),
        "git": _git_info(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "provider": {
            "provider_id": provider_metadata.provider_id,
            "provider_version": provider_metadata.provider_version,
            "template_format": provider_metadata.template_format,
            "template_version": provider_metadata.template_version,
            "sdk_name": provider_metadata.sdk_name,
            "service_url": os.getenv(SOURCEAFIS_SERVICE_URL_ENV, ""),
            "metadata": provider_metadata.metadata,
        },
        "score_semantics": {
            "raw_score_name": "SourceAFIS raw similarity score",
            "higher_score_more_similar": True,
            "normalization": "none",
            "calibration": "thresholds selected on VAL only and applied unchanged to TEST",
        },
        "target_fars": [float(x) for x in target_fars],
        "datasets": dataset_statuses,
        "template_cache": {
            "path": str(cache_dir),
            "contains_template_bytes": True,
            "summary_reports_expose_template_bytes": False,
        },
        "outputs": {key: str(value) for key, value in output_paths.items()},
        "output_schema": output_schema(),
        "total_runtime_s": float(total_runtime_s),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _try_write_sift_comparison(
    *,
    outdir: Path,
    metrics: pd.DataFrame,
    repo_root: Path,
) -> bool:
    sift_metrics_path = (
        repo_root
        / "artifacts"
        / "reports"
        / "benchmark"
        / "sift_plain_roll_v2_external_validation"
        / "per_dataset_metrics.csv"
    )
    if not sift_metrics_path.exists() or metrics.empty:
        return False

    sift = pd.read_csv(sift_metrics_path)
    rows: list[dict[str, Any]] = []
    sourceafis_test = metrics[metrics["split"] == "test"].copy()
    sift_test = sift[sift["split"].astype(str).str.lower() == "test"].copy()
    for _, src in sourceafis_test.iterrows():
        matches = sift_test[
            (sift_test["dataset"] == src["dataset"])
            & np.isclose(sift_test["target_far"].astype(float), float(src["target_far"]))
            & (sift_test["method"] == "sift_plain_roll_v2")
            & (sift_test["variant"] == "official_score")
        ]
        sift_row = matches.iloc[0] if not matches.empty else None
        rows.append(
            {
                "dataset": src["dataset"],
                "target_far": float(src["target_far"]),
                "sourceafis_threshold": float(src["threshold"]),
                "sourceafis_tar": float(src["tar"]),
                "sourceafis_far": float(src["far"]),
                "sourceafis_auc": float(src["auc"]),
                "sourceafis_eer": float(src["eer"]),
                "sift_v2_threshold": float(sift_row["threshold"]) if sift_row is not None else float("nan"),
                "sift_v2_tar": float(sift_row["tar"]) if sift_row is not None else float("nan"),
                "sift_v2_far": float(sift_row["far"]) if sift_row is not None else float("nan"),
                "sift_v2_auc": float(sift_row["auc"]) if sift_row is not None else float("nan"),
                "sift_v2_eer": float(sift_row["eer"]) if sift_row is not None else float("nan"),
                "tar_delta_sourceafis_minus_sift_v2": (
                    float(src["tar"]) - float(sift_row["tar"]) if sift_row is not None else float("nan")
                ),
                "far_delta_sourceafis_minus_sift_v2": (
                    float(src["far"]) - float(sift_row["far"]) if sift_row is not None else float("nan")
                ),
                "sift_source": str(sift_metrics_path),
            }
        )
    comparison = pd.DataFrame(rows)
    csv_path = outdir / "sourceafis_vs_sift_v2_comparison.csv"
    md_path = outdir / "sourceafis_vs_sift_v2_comparison.md"
    comparison.to_csv(csv_path, index=False)

    lines = [
        "# SourceAFIS vs SIFT Plain/Roll v2",
        "",
        f"SIFT v2 source: `{sift_metrics_path}`",
        "",
        "| dataset | target FAR | SourceAFIS TAR | SourceAFIS FAR | SIFT v2 TAR | SIFT v2 FAR | TAR delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.sort_values(["dataset", "target_far"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {_fmt_pct(row['target_far'])} | {_fmt_pct(row['sourceafis_tar'])} | "
            f"{_fmt_pct(row['sourceafis_far'])} | {_fmt_pct(row['sift_v2_tar'])} | "
            f"{_fmt_pct(row['sift_v2_far'])} | {_fmt_pct(row['tar_delta_sourceafis_minus_sift_v2'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def run_benchmark(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    outdir: Path = DEFAULT_OUTDIR,
    target_fars: tuple[float, ...] = TARGET_FARS,
    limit_per_split: int = 0,
    engine: FingerprintEngine | None = None,
    require_enabled_env: bool = True,
    repo_root: Path = REPO_ROOT,
    template_cache_dir: Path | None = None,
) -> dict[str, Path]:
    start = time.perf_counter()
    output = validate_output_directory(outdir, repo_root=repo_root)
    output.mkdir(parents=True, exist_ok=True)
    selected_engine, provider_metadata = ensure_sourceafis_available(
        engine,
        require_enabled_env=require_enabled_env,
    )

    service_url = str(os.getenv(SOURCEAFIS_SERVICE_URL_ENV) or provider_metadata.metadata.get("service_url") or "")
    cache_dir = template_cache_dir or (output / "template_cache")
    cache = TemplateCache(
        parse_file_uri(cache_dir, repo_root=repo_root),
        provider_metadata=provider_metadata,
        service_url=service_url,
        repo_root=repo_root,
    )

    dataset_statuses: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    failure_rows: list[dict[str, Any]] = []
    latency_events: list[dict[str, Any]] = []

    for dataset in datasets:
        for split in splits:
            pairs, status = load_plain_roll_pairs(
                dataset,
                split,
                repo_root=repo_root,
                limit=int(limit_per_split),
            )
            dataset_statuses.append(status)
            if pairs.empty:
                continue
            scored, failures, events = score_pairs(pairs, engine=selected_engine, cache=cache)
            score_frames.append(scored)
            failure_rows.extend(failures)
            latency_events.extend(events)

    if not score_frames:
        raise SourceAfisBenchmarkError("No compatible plain-vs-roll pairs were scored.")

    scores = pd.concat(score_frames, ignore_index=True, sort=False)
    thresholds = build_threshold_table(scores, tuple(float(x) for x in target_fars))
    metrics = build_metrics_table(scores, thresholds)
    latency = build_latency_summary(latency_events)
    failures = pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS)

    paths = {
        "scores_val": output / "sourceafis_plain_roll_scores_val.csv",
        "scores_test": output / "sourceafis_plain_roll_scores_test.csv",
        "thresholds": output / "sourceafis_plain_roll_thresholds.csv",
        "metrics": output / "sourceafis_plain_roll_metrics.csv",
        "latency_summary": output / "sourceafis_plain_roll_latency_summary.csv",
        "failures": output / "sourceafis_plain_roll_failures.csv",
        "summary": output / "sourceafis_plain_roll_summary.md",
        "manifest": output / "sourceafis_plain_roll_manifest.json",
    }

    _write_csv(paths["scores_val"], scores[scores["split"] == "val"], SCORES_COLUMNS)
    _write_csv(paths["scores_test"], scores[scores["split"] == "test"], SCORES_COLUMNS)
    _write_csv(paths["thresholds"], thresholds, THRESHOLD_COLUMNS)
    _write_csv(paths["metrics"], metrics, METRICS_COLUMNS)
    _write_csv(paths["latency_summary"], latency, LATENCY_COLUMNS)
    _write_csv(paths["failures"], failures, FAILURE_COLUMNS)

    comparison_created = _try_write_sift_comparison(outdir=output, metrics=metrics, repo_root=repo_root)
    if comparison_created:
        paths["comparison_csv"] = output / "sourceafis_vs_sift_v2_comparison.csv"
        paths["comparison_markdown"] = output / "sourceafis_vs_sift_v2_comparison.md"

    total_runtime_s = time.perf_counter() - start
    paths["summary"].write_text(
        render_summary_markdown(
            metrics=metrics,
            thresholds=thresholds,
            latency=latency,
            failures=failures,
            dataset_statuses=dataset_statuses,
            total_runtime_s=total_runtime_s,
            output_paths=paths,
            comparison_created=comparison_created,
        ),
        encoding="utf-8",
    )
    write_manifest(
        paths["manifest"],
        provider_metadata=provider_metadata,
        dataset_statuses=dataset_statuses,
        output_paths=paths,
        target_fars=tuple(float(x) for x in target_fars),
        total_runtime_s=total_runtime_s,
        repo_root=repo_root,
        cache_dir=cache.cache_dir,
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark sourceafis_open on NIST plain-vs-roll positive/negative VAL/TEST protocols."
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--target_far", type=float, nargs="*", default=list(TARGET_FARS))
    parser.add_argument(
        "--limit_per_split",
        type=int,
        default=0,
        help="Optional smoke-test cap per dataset/split. Default 0 evaluates all compatible pairs.",
    )
    parser.add_argument(
        "--template_cache_dir",
        default="",
        help="Optional template cache directory. Defaults to <outdir>/template_cache.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = tuple(item.strip() for item in str(args.datasets).split(",") if item.strip())
    splits = tuple(item.strip().lower() for item in str(args.splits).split(",") if item.strip())
    template_cache_dir = parse_file_uri(args.template_cache_dir) if str(args.template_cache_dir).strip() else None
    try:
        paths = run_benchmark(
            datasets=datasets,
            splits=splits,
            outdir=parse_file_uri(args.outdir),
            target_fars=tuple(float(x) for x in args.target_far),
            limit_per_split=int(args.limit_per_split),
            template_cache_dir=template_cache_dir,
        )
    except (FingerprintEngineError, SourceAfisBenchmarkError) as exc:
        print(f"SourceAFIS Plain/Roll benchmark unavailable: {exc}", file=sys.stderr)
        return 2

    print("Wrote SourceAFIS Plain/Roll benchmark artifacts:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
