from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.candidate_source_resolver import (  # noqa: E402
    SOURCE_BROWSER_CATALOG,
    SOURCE_CANDIDATE_SOURCE_MISSING,
    SOURCE_DEMO_CATALOG,
    SOURCE_NO_CANDIDATE_SOURCE,
    SOURCE_TEST_FIXTURE,
    CandidateSourceResolver,
)
from apps.api.identification_service import IdentificationService  # noqa: E402
from apps.api.method_registry import ApiMethodRegistry, MethodRegistryError, load_api_method_registry  # noqa: E402
from src.fpbench.identification.secure_split_store import SecureSplitFingerprintStore  # noqa: E402


DEFAULT_DATASET = "nist_sd300b"
DEFAULT_METHODS = ("classic_gftt_orb", "minutiae", "harris", "sift", "dl")
OPTIONAL_METHODS = ("vit",)
DEFAULT_LIMIT = 1000
DEFAULT_SHORTLIST_SIZE = 25
DEFAULT_SCORE_EPSILON = 1e-6
DEFAULT_SEED = 1337
DEFAULT_RERANK_POLICY = "top1"
RERANK_POLICIES = ("full", "top1", "none")

REQUIRED_MANIFEST_COLUMNS = ("dataset", "capture", "subject_id", "impression", "ppi", "frgp", "path")
BASE_TABLE_NAMES = {
    "person_directory",
    "identity_map",
    "raw_fingerprints",
    "feature_vectors",
    "method_retrieval_vectors",
}
PREFIX_RE = re.compile(r"[^a-zA-Z0-9_]+")
SAFE_PREFIX_RE = re.compile(r"(?=.*[a-zA-Z0-9])[a-zA-Z0-9_]+_$")
RANDOM_ID_SAFE_RE = re.compile(r"[^a-z0-9_]+")

AVAILABLE_CANDIDATE_SOURCE_STATUSES = {
    SOURCE_DEMO_CATALOG,
    SOURCE_BROWSER_CATALOG,
    SOURCE_TEST_FIXTURE,
    "legacy_db_image_bytes",
}
UNAVAILABLE_CANDIDATE_SOURCE_STATUSES = {
    "",
    SOURCE_NO_CANDIDATE_SOURCE,
    SOURCE_CANDIDATE_SOURCE_MISSING,
    "no_raw_metadata",
    "legacy_db_image_bytes_unavailable",
}

SUMMARY_COLUMNS = [
    "dataset",
    "table_prefix",
    "method",
    "rerank_policy",
    "n_selected",
    "n_enrolled",
    "n_queries",
    "enroll_error_count",
    "query_error_count",
    "top1_self_match_count",
    "top1_self_match_rate",
    "retrieval_top1_self_match_count",
    "retrieval_top1_self_match_rate",
    "final_top1_self_match_count",
    "final_top1_self_match_rate",
    "self_in_shortlist_count",
    "self_in_shortlist_rate",
    "mean_self_rank",
    "p95_self_rank",
    "mean_retrieval_rank_self",
    "mean_final_rank_self",
    "exact_self_vector_score_count",
    "exact_self_vector_score_rate_epsilon",
    "mean_self_retrieval_score",
    "min_self_retrieval_score",
    "mean_top1_retrieval_score",
    "mean_self_rerank_score",
    "mean_top1_rerank_score",
    "mean_probe_embed_ms",
    "mean_shortlist_scan_ms",
    "mean_rerank_ms",
    "mean_total_query_ms",
    "p95_total_query_ms",
    "reranked_candidate_count_mean",
    "skipped_rerank_candidate_count_mean",
    "rerank_performed_rate",
    "candidate_source_available_rate",
    "notes",
]

ENROLLMENT_COLUMNS = [
    "selected_index",
    "manifest_row_index",
    "expected_random_id",
    "random_id",
    "enrolled",
    "image_path",
    "subject_id",
    "frgp",
    "impression",
    "capture",
    "full_name",
    "national_id",
    "elapsed_ms",
    "vector_methods",
    "image_sha256",
    "created_at",
    "error_type",
    "error",
]

QUERY_COLUMNS = [
    "method",
    "rerank_policy",
    "selected_index",
    "manifest_row_index",
    "expected_random_id",
    "image_path",
    "subject_id",
    "frgp",
    "impression",
    "capture",
    "candidate_pool_size",
    "shortlist_size",
    "self_in_shortlist",
    "retrieval_top1_random_id",
    "retrieval_top1_is_self",
    "retrieval_rank_self",
    "final_top1_random_id",
    "final_top1_is_self",
    "final_rank_self",
    "self_rank",
    "top1_random_id",
    "top1_is_self",
    "retrieval_score_self",
    "top1_retrieval_score",
    "rerank_score_self",
    "top1_rerank_score",
    "decision",
    "decision_status",
    "decision_basis",
    "rerank_status",
    "rerank_performed",
    "reranked_candidate_count",
    "skipped_rerank_candidate_count",
    "candidate_source_status",
    "candidate_source_available",
    "top1_candidate_source_status",
    "self_candidate_source_status",
    "probe_embed_ms",
    "shortlist_scan_ms",
    "rerank_ms",
    "total_query_ms",
]

FAILURE_COLUMNS = [
    "method",
    "selected_index",
    "manifest_row_index",
    "expected_random_id",
    "image_path",
    "subject_id",
    "frgp",
    "impression",
    "capture",
    "error_type",
    "error",
    "traceback",
]


@dataclass(frozen=True)
class ManifestExperimentRow:
    selected_index: int
    manifest_row_index: int
    dataset: str
    capture: str
    subject_id: str
    impression: str
    ppi: str
    frgp: str
    image_path: Path

    @property
    def full_name(self) -> str:
        return (
            f"Experiment {self.dataset} {self.manifest_row_index} "
            f"{self.subject_id} {self.frgp} {self.impression}"
        )

    @property
    def national_id(self) -> str:
        return synthetic_national_id(self.manifest_row_index)

    @property
    def expected_random_id(self) -> str:
        payload = "|".join(
            [
                self.dataset,
                str(self.manifest_row_index),
                self.subject_id,
                self.frgp,
                self.impression,
                str(self.image_path),
            ]
        )
        digest = _short_digest(payload)
        dataset_slug = _safe_random_id_part(self.dataset)
        return f"selfmatch_{dataset_slug}_{self.manifest_row_index:06d}_{digest}"

    def expected_mapping(self) -> dict[str, Any]:
        return {
            "selected_index": self.selected_index,
            "manifest_row_index": self.manifest_row_index,
            "expected_random_id": self.expected_random_id,
            "image_path": str(self.image_path),
            "subject_id": self.subject_id,
            "frgp": self.frgp,
            "impression": self.impression,
            "capture": self.capture,
            "full_name": self.full_name,
            "national_id": self.national_id,
        }


@dataclass(frozen=True)
class ManifestSelectionReport:
    manifest_path: Path
    total_rows: int
    missing_path_count: int
    capture_filtered_count: int
    valid_row_count: int
    selected_count: int
    limit: int
    seed: int
    capture_filter: str | None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["manifest_path"] = str(self.manifest_path)
        return payload


def _short_digest(payload: str) -> str:
    import hashlib

    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _safe_random_id_part(value: str) -> str:
    cleaned = RANDOM_ID_SAFE_RE.sub("_", str(value).strip().lower()).strip("_")
    return cleaned or "dataset"


def normalize_table_prefix(raw: str | None) -> str:
    prefix = PREFIX_RE.sub("_", str(raw or "").strip().lower())
    if prefix and not prefix.endswith("_"):
        prefix += "_"
    return prefix


def default_table_prefix(timestamp: str | None = None) -> str:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"self_match_exp_{stamp}_"


def ensure_safe_reset_prefix(table_prefix: str | None) -> str:
    prefix = normalize_table_prefix(table_prefix)
    if not prefix:
        raise ValueError("--reset-prefix requires a non-empty experiment table prefix.")
    if len(prefix) < 4:
        raise ValueError(f"--reset-prefix refused suspiciously short table prefix: {prefix!r}")
    if not SAFE_PREFIX_RE.fullmatch(prefix):
        raise ValueError(f"--reset-prefix refused unsafe table prefix: {prefix!r}")
    if prefix.rstrip("_") in BASE_TABLE_NAMES:
        raise ValueError(f"--reset-prefix refused production table-like prefix: {prefix!r}")
    for base_name in BASE_TABLE_NAMES:
        if base_name.startswith(prefix) or prefix.startswith(f"{base_name}_"):
            raise ValueError(f"--reset-prefix refused production table-like prefix: {prefix!r}")
    return prefix


def synthetic_national_id(row_index: int) -> str:
    value = int(row_index)
    if value <= 0:
        raise ValueError("row_index must be 1-based and positive")
    if value > 999_999_999:
        raise ValueError("row_index is too large for a deterministic 9-digit synthetic national_id")
    return f"{value:09d}"


def resolve_manifest_path(raw_path: str, *, repo_root: Path = REPO_ROOT) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        return repo_root / "__missing_manifest_path__"
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def default_manifest_for_dataset(dataset: str, *, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / "data" / "manifests" / dataset / "manifest.csv"


def load_manifest_selection(
    *,
    dataset: str = DEFAULT_DATASET,
    manifest_path: str | Path | None = None,
    repo_root: Path = REPO_ROOT,
    limit: int = DEFAULT_LIMIT,
    seed: int = DEFAULT_SEED,
    capture_filter: str | None = None,
) -> tuple[list[ManifestExperimentRow], ManifestSelectionReport]:
    manifest = Path(manifest_path) if manifest_path is not None else default_manifest_for_dataset(dataset, repo_root=repo_root)
    if not manifest.is_absolute():
        manifest = (repo_root / manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest.csv: {manifest}")

    valid_rows: list[ManifestExperimentRow] = []
    total_rows = 0
    missing_path_count = 0
    capture_filtered_count = 0
    capture_filter_norm = str(capture_filter).strip().lower() if capture_filter else None

    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in fieldnames]
        if missing_columns:
            raise ValueError(f"Manifest {manifest} is missing required columns: {missing_columns}")

        for row_index, row in enumerate(reader, start=1):
            total_rows += 1
            capture = str(row.get("capture") or "").strip() or "plain"
            if capture_filter_norm and capture.lower() != capture_filter_norm:
                capture_filtered_count += 1
                continue

            image_path = resolve_manifest_path(str(row.get("path") or ""), repo_root=repo_root)
            if not image_path.is_file():
                missing_path_count += 1
                continue

            valid_rows.append(
                ManifestExperimentRow(
                    selected_index=0,
                    manifest_row_index=row_index,
                    dataset=str(row.get("dataset") or dataset).strip() or dataset,
                    capture=capture,
                    subject_id=str(row.get("subject_id") or "").strip(),
                    impression=str(row.get("impression") or "").strip(),
                    ppi=str(row.get("ppi") or "").strip(),
                    frgp=str(row.get("frgp") or "").strip(),
                    image_path=image_path,
                )
            )

    limit_value = int(limit)
    selected = list(valid_rows)
    if limit_value > 0 and len(selected) > limit_value:
        rng = random.Random(int(seed))
        rng.shuffle(selected)
        selected = selected[:limit_value]

    selected = [
        ManifestExperimentRow(
            selected_index=index,
            manifest_row_index=row.manifest_row_index,
            dataset=row.dataset,
            capture=row.capture,
            subject_id=row.subject_id,
            impression=row.impression,
            ppi=row.ppi,
            frgp=row.frgp,
            image_path=row.image_path,
        )
        for index, row in enumerate(selected, start=1)
    ]
    report = ManifestSelectionReport(
        manifest_path=manifest,
        total_rows=total_rows,
        missing_path_count=missing_path_count,
        capture_filtered_count=capture_filtered_count,
        valid_row_count=len(valid_rows),
        selected_count=len(selected),
        limit=limit_value,
        seed=int(seed),
        capture_filter=capture_filter,
    )
    return selected, report


def resolve_experiment_methods(
    raw_methods: str | Sequence[str] | None = None,
    *,
    include_vit: bool = False,
    registry: ApiMethodRegistry | None = None,
) -> list[str]:
    resolved_registry = registry or load_api_method_registry()
    if raw_methods is None:
        requested = list(DEFAULT_METHODS)
    elif isinstance(raw_methods, str):
        stripped = raw_methods.strip()
        if not stripped or stripped.lower() in {"default", "advisor", "advisor_default"}:
            requested = list(DEFAULT_METHODS)
        else:
            requested = [item.strip() for item in stripped.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in raw_methods if str(item).strip()]
        if not requested:
            requested = list(DEFAULT_METHODS)

    if include_vit:
        requested.extend(OPTIONAL_METHODS)

    methods: list[str] = []
    seen: set[str] = set()
    for method in requested:
        try:
            retrieval = resolved_registry.resolve_retrieval_method(method)
            rerank = resolved_registry.resolve_rerank_method(method)
        except MethodRegistryError as exc:
            raise ValueError(str(exc)) from exc
        canonical = retrieval.canonical_api_name
        if canonical != rerank.canonical_api_name:
            raise ValueError(
                f"Method {method!r} resolved inconsistently for retrieval/rerank: "
                f"{canonical!r} vs {rerank.canonical_api_name!r}"
            )
        if canonical in seen:
            continue
        methods.append(canonical)
        seen.add(canonical)
    return methods


def _format_command(argv: Sequence[str]) -> str:
    parts = [sys.executable, *argv]
    try:
        return subprocess.list2cmdline([str(part) for part in parts])
    except Exception:
        return " ".join(str(part) for part in parts)


def _run_git(args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def collect_git_info() -> dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": commit or "unknown",
        "dirty": bool(status),
        "status_porcelain": status or "",
    }


def reset_prefixed_experiment_tables(
    *,
    table_prefix: str,
    database_url: str | None = None,
    identity_database_url: str | None = None,
) -> list[str]:
    safe_prefix = ensure_safe_reset_prefix(table_prefix)
    store = SecureSplitFingerprintStore.for_inspection(
        database_url=database_url,
        identity_database_url=identity_database_url,
        table_prefix=safe_prefix,
    )
    if store.table_prefix != safe_prefix:
        raise ValueError(f"Resolved table prefix mismatch: {safe_prefix!r} -> {store.table_prefix!r}")

    dropped: list[str] = []
    biometric_tables = [
        store.generic_vector_table,
        store.vector_table,
        store.raw_table,
    ]
    if not store.dual_database_enabled:
        biometric_tables.append(store.identity_table)
    biometric_tables.append(store.person_table)

    with store._connect_biometric(autocommit=True) as conn:  # noqa: SLF001 - research maintenance script.
        with conn.cursor() as cur:
            for table_name in biometric_tables:
                _assert_table_name_has_prefix(table_name, safe_prefix)
                cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                dropped.append(f"biometric_db.{table_name}")

    if store.dual_database_enabled:
        with store._connect_identity(autocommit=True) as conn:  # noqa: SLF001 - research maintenance script.
            with conn.cursor() as cur:
                _assert_table_name_has_prefix(store.identity_table, safe_prefix)
                cur.execute(f"DROP TABLE IF EXISTS {store.identity_table} CASCADE")
                dropped.append(f"identity_db.{store.identity_table}")
    return dropped


def _assert_table_name_has_prefix(table_name: str, prefix: str) -> None:
    if not str(table_name).startswith(prefix):
        raise ValueError(f"Refusing to reset unprefixed table name {table_name!r} for prefix {prefix!r}")


def _candidate_source_available(status: str | None) -> bool:
    value = str(status or "")
    if value in AVAILABLE_CANDIDATE_SOURCE_STATUSES:
        return True
    if value in UNAVAILABLE_CANDIDATE_SOURCE_STATUSES:
        return False
    return not value.startswith("skipped_") and value not in {"unknown", "none"}


def identify_kwargs_for_rerank_policy(rerank_policy: str) -> dict[str, Any]:
    policy = str(rerank_policy or DEFAULT_RERANK_POLICY).strip().lower()
    if policy == "full":
        return {"skip_rerank": False, "rerank_limit": None}
    if policy == "top1":
        return {"skip_rerank": False, "rerank_limit": 1}
    if policy == "none":
        return {"skip_rerank": True, "rerank_limit": 0}
    raise ValueError(f"Unsupported rerank_policy={rerank_policy!r}; expected one of {RERANK_POLICIES}")


def enroll_selected_rows(
    *,
    service: IdentificationService,
    resolver: CandidateSourceResolver,
    rows: Sequence[ManifestExperimentRow],
    methods: Sequence[str],
    fail_fast: bool,
) -> tuple[list[dict[str, Any]], list[ManifestExperimentRow]]:
    enrollment_rows: list[dict[str, Any]] = []
    enrolled_rows: list[ManifestExperimentRow] = []
    total = len(rows)
    for row in rows:
        print(f"[enroll] {row.selected_index}/{total} {row.image_path.name}")
        t0 = time.perf_counter()
        base = {
            "selected_index": row.selected_index,
            "manifest_row_index": row.manifest_row_index,
            "expected_random_id": row.expected_random_id,
            "image_path": str(row.image_path),
            "subject_id": row.subject_id,
            "frgp": row.frgp,
            "impression": row.impression,
            "capture": row.capture,
            "full_name": row.full_name,
            "national_id": row.national_id,
        }
        try:
            receipt = service.enroll_from_path(
                path=str(row.image_path),
                full_name=row.full_name,
                national_id=row.national_id,
                capture=row.capture,
                vector_methods=list(methods),
                replace_existing=True,
                random_id=row.expected_random_id,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            resolver.register_test_source(receipt.random_id, row.image_path, capture=row.capture)
            enrolled_rows.append(row)
            enrollment_rows.append(
                {
                    **base,
                    "random_id": receipt.random_id,
                    "enrolled": True,
                    "elapsed_ms": elapsed_ms,
                    "vector_methods": ",".join(receipt.vector_methods),
                    "image_sha256": receipt.image_sha256,
                    "created_at": receipt.created_at,
                    "error_type": "",
                    "error": "",
                }
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            enrollment_rows.append(
                {
                    **base,
                    "random_id": "",
                    "enrolled": False,
                    "elapsed_ms": elapsed_ms,
                    "vector_methods": ",".join(methods),
                    "image_sha256": "",
                    "created_at": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[enroll] ERROR row={row.selected_index}: {type(exc).__name__}: {exc}")
            if fail_fast:
                raise
    return enrollment_rows, enrolled_rows


def query_self_matches_for_method(
    *,
    service: IdentificationService,
    method: str,
    rows: Sequence[ManifestExperimentRow],
    shortlist_size: int,
    fail_fast: bool,
    rerank_policy: str = DEFAULT_RERANK_POLICY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    total = len(rows)
    policy = str(rerank_policy or DEFAULT_RERANK_POLICY).strip().lower()
    identify_rerank_kwargs = identify_kwargs_for_rerank_policy(policy)
    for row in rows:
        print(f"[query:{method}:{policy}] {row.selected_index}/{total} {row.image_path.name}")
        try:
            result = service.identify_from_path(
                path=str(row.image_path),
                capture=row.capture,
                retrieval_method=method,
                rerank_method=method,
                shortlist_size=shortlist_size,
                **identify_rerank_kwargs,
            )
            top = result.top_candidate
            retrieval_top = min(
                result.candidates,
                key=lambda candidate: getattr(candidate, "retrieval_rank", candidate.rank),
                default=None,
            )
            self_candidate = next(
                (candidate for candidate in result.candidates if candidate.random_id == row.expected_random_id),
                None,
            )
            candidate_source_status = (
                self_candidate.candidate_source_status
                if self_candidate is not None
                else (top.candidate_source_status if top is not None else "")
            )
            rerank_performed = int(result.rerank_summary.get("performed_count") or 0) > 0
            latency = result.latency_ms
            retrieval_rank_self = (
                getattr(self_candidate, "retrieval_rank", self_candidate.rank)
                if self_candidate is not None
                else ""
            )
            final_rank_self = self_candidate.rank if self_candidate is not None else ""
            retrieval_top1_random_id = retrieval_top.random_id if retrieval_top is not None else ""
            final_top1_random_id = top.random_id if top is not None else ""
            retrieval_top1_is_self = bool(
                retrieval_top is not None and retrieval_top.random_id == row.expected_random_id
            )
            final_top1_is_self = bool(top is not None and top.random_id == row.expected_random_id)
            reranked_candidate_count = int(result.rerank_summary.get("performed_count") or 0)
            skipped_rerank_candidate_count = int(result.rerank_summary.get("skipped_count") or 0)
            query_rows.append(
                {
                    "method": method,
                    "rerank_policy": policy,
                    "selected_index": row.selected_index,
                    "manifest_row_index": row.manifest_row_index,
                    "expected_random_id": row.expected_random_id,
                    "image_path": str(row.image_path),
                    "subject_id": row.subject_id,
                    "frgp": row.frgp,
                    "impression": row.impression,
                    "capture": row.capture,
                    "candidate_pool_size": result.candidate_pool_size,
                    "shortlist_size": result.shortlist_size,
                    "self_in_shortlist": self_candidate is not None,
                    "retrieval_top1_random_id": retrieval_top1_random_id,
                    "retrieval_top1_is_self": retrieval_top1_is_self,
                    "retrieval_rank_self": retrieval_rank_self,
                    "final_top1_random_id": final_top1_random_id,
                    "final_top1_is_self": final_top1_is_self,
                    "final_rank_self": final_rank_self,
                    "self_rank": final_rank_self,
                    "top1_random_id": final_top1_random_id,
                    "top1_is_self": final_top1_is_self,
                    "retrieval_score_self": (
                        self_candidate.retrieval_score if self_candidate is not None else ""
                    ),
                    "top1_retrieval_score": top.retrieval_score if top is not None else "",
                    "rerank_score_self": (
                        self_candidate.rerank_score
                        if self_candidate is not None and self_candidate.rerank_score is not None
                        else ""
                    ),
                    "top1_rerank_score": (
                        top.rerank_score if top is not None and top.rerank_score is not None else ""
                    ),
                    "decision": result.decision,
                    "decision_status": result.decision_status,
                    "decision_basis": result.decision_basis,
                    "rerank_status": result.rerank_status,
                    "rerank_performed": rerank_performed,
                    "reranked_candidate_count": reranked_candidate_count,
                    "skipped_rerank_candidate_count": skipped_rerank_candidate_count,
                    "candidate_source_status": candidate_source_status,
                    "candidate_source_available": _candidate_source_available(candidate_source_status),
                    "top1_candidate_source_status": top.candidate_source_status if top is not None else "",
                    "self_candidate_source_status": (
                        self_candidate.candidate_source_status if self_candidate is not None else ""
                    ),
                    "probe_embed_ms": latency.get("probe_embed_ms", ""),
                    "shortlist_scan_ms": latency.get("shortlist_scan_ms", ""),
                    "rerank_ms": latency.get("rerank_ms", ""),
                    "total_query_ms": latency.get("total_ms", ""),
                }
            )
        except Exception as exc:
            failure = {
                "method": method,
                "selected_index": row.selected_index,
                "manifest_row_index": row.manifest_row_index,
                "expected_random_id": row.expected_random_id,
                "image_path": str(row.image_path),
                "subject_id": row.subject_id,
                "frgp": row.frgp,
                "impression": row.impression,
                "capture": row.capture,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failure_rows.append(failure)
            print(f"[query:{method}] ERROR row={row.selected_index}: {type(exc).__name__}: {exc}")
            if fail_fast:
                raise
    return query_rows, failure_rows


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _nearest_rank_p95(values: Sequence[float]) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    index = max(0, math.ceil(0.95 * len(clean)) - 1)
    return clean[index]


def _rate(count: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(count) / float(denominator)


def summarize_method_results(
    *,
    dataset: str,
    table_prefix: str,
    method: str,
    n_selected: int,
    n_enrolled: int,
    enroll_error_count: int,
    query_rows: Sequence[Mapping[str, Any]],
    query_error_count: int,
    rerank_policy: str = DEFAULT_RERANK_POLICY,
    score_epsilon: float = DEFAULT_SCORE_EPSILON,
) -> dict[str, Any]:
    n_queries = len(query_rows)
    retrieval_top1_self_match_count = sum(
        1 for row in query_rows if _as_bool(row.get("retrieval_top1_is_self"))
    )
    final_top1_self_match_count = sum(
        1
        for row in query_rows
        if _as_bool(row.get("final_top1_is_self", row.get("top1_is_self")))
    )
    top1_self_match_count = final_top1_self_match_count
    self_in_shortlist_count = sum(1 for row in query_rows if _as_bool(row.get("self_in_shortlist")))
    final_self_ranks = [
        value
        for value in (
            _as_float(row.get("final_rank_self", row.get("self_rank"))) for row in query_rows
        )
        if value is not None
    ]
    retrieval_self_ranks = [
        value
        for value in (_as_float(row.get("retrieval_rank_self")) for row in query_rows)
        if value is not None
    ]
    self_retrieval_scores = [
        value
        for value in (_as_float(row.get("retrieval_score_self")) for row in query_rows)
        if value is not None
    ]
    top1_retrieval_scores = [
        value
        for value in (_as_float(row.get("top1_retrieval_score")) for row in query_rows)
        if value is not None
    ]
    self_rerank_scores = [
        value
        for value in (_as_float(row.get("rerank_score_self")) for row in query_rows)
        if value is not None
    ]
    top1_rerank_scores = [
        value
        for value in (_as_float(row.get("top1_rerank_score")) for row in query_rows)
        if value is not None
    ]
    probe_embed_ms = [_as_float(row.get("probe_embed_ms")) for row in query_rows]
    shortlist_scan_ms = [_as_float(row.get("shortlist_scan_ms")) for row in query_rows]
    rerank_ms = [_as_float(row.get("rerank_ms")) for row in query_rows]
    reranked_candidate_counts = [_as_float(row.get("reranked_candidate_count")) for row in query_rows]
    skipped_rerank_candidate_counts = [
        _as_float(row.get("skipped_rerank_candidate_count")) for row in query_rows
    ]
    total_query_ms = [
        value
        for value in (_as_float(row.get("total_query_ms")) for row in query_rows)
        if value is not None
    ]
    exact_self_vector_score_count = sum(
        1
        for score in self_retrieval_scores
        if abs(1.0 - float(score)) <= float(score_epsilon)
    )
    rerank_performed_count = sum(1 for row in query_rows if _as_bool(row.get("rerank_performed")))
    candidate_source_available_count = sum(
        1 for row in query_rows if _as_bool(row.get("candidate_source_available"))
    )

    notes: list[str] = []
    if n_queries == 0:
        notes.append("no successful queries")
    if enroll_error_count:
        notes.append(f"{enroll_error_count} enrollment errors")
    if query_error_count:
        notes.append(f"{query_error_count} query errors")
    if n_queries and top1_self_match_count < n_queries:
        notes.append("not every successful self-query ranked itself first")
    if n_queries and rerank_performed_count < n_queries:
        notes.append("rerank was skipped or failed for at least one query")
    if n_queries and candidate_source_available_count < n_queries:
        notes.append("candidate source unavailable for at least one query")
    notes.append(f"exact vector score epsilon={score_epsilon:g}")

    return {
        "dataset": dataset,
        "table_prefix": table_prefix,
        "method": method,
        "rerank_policy": rerank_policy,
        "n_selected": int(n_selected),
        "n_enrolled": int(n_enrolled),
        "n_queries": int(n_queries),
        "enroll_error_count": int(enroll_error_count),
        "query_error_count": int(query_error_count),
        "top1_self_match_count": int(top1_self_match_count),
        "top1_self_match_rate": _rate(top1_self_match_count, n_queries),
        "retrieval_top1_self_match_count": int(retrieval_top1_self_match_count),
        "retrieval_top1_self_match_rate": _rate(retrieval_top1_self_match_count, n_queries),
        "final_top1_self_match_count": int(final_top1_self_match_count),
        "final_top1_self_match_rate": _rate(final_top1_self_match_count, n_queries),
        "self_in_shortlist_count": int(self_in_shortlist_count),
        "self_in_shortlist_rate": _rate(self_in_shortlist_count, n_queries),
        "mean_self_rank": _mean(final_self_ranks),
        "p95_self_rank": _nearest_rank_p95(final_self_ranks),
        "mean_retrieval_rank_self": _mean(retrieval_self_ranks),
        "mean_final_rank_self": _mean(final_self_ranks),
        "exact_self_vector_score_count": int(exact_self_vector_score_count),
        "exact_self_vector_score_rate_epsilon": _rate(exact_self_vector_score_count, n_queries),
        "mean_self_retrieval_score": _mean(self_retrieval_scores),
        "min_self_retrieval_score": min(self_retrieval_scores) if self_retrieval_scores else None,
        "mean_top1_retrieval_score": _mean(top1_retrieval_scores),
        "mean_self_rerank_score": _mean(self_rerank_scores),
        "mean_top1_rerank_score": _mean(top1_rerank_scores),
        "mean_probe_embed_ms": _mean([value for value in probe_embed_ms if value is not None]),
        "mean_shortlist_scan_ms": _mean([value for value in shortlist_scan_ms if value is not None]),
        "mean_rerank_ms": _mean([value for value in rerank_ms if value is not None]),
        "mean_total_query_ms": _mean(total_query_ms),
        "p95_total_query_ms": _nearest_rank_p95(total_query_ms),
        "reranked_candidate_count_mean": _mean(
            [value for value in reranked_candidate_counts if value is not None]
        ),
        "skipped_rerank_candidate_count_mean": _mean(
            [value for value in skipped_rerank_candidate_counts if value is not None]
        ),
        "rerank_performed_rate": _rate(rerank_performed_count, n_queries),
        "candidate_source_available_rate": _rate(candidate_source_available_count, n_queries),
        "notes": "; ".join(notes),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _format_rate(value: object) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number * 100.0:.2f}%"


def _format_number(value: object, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def render_summary_markdown(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    command: str,
    table_prefix: str,
    db_layout: Mapping[str, Any],
    git_info: Mapping[str, Any],
    output_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Identification Self-Match Repeatability Experiment")
    lines.append("")
    lines.append(
        "This experiment is a 1:N identification self-match repeatability experiment, not "
        "the existing 1:1 verification benchmark. Each enrolled image is treated as its "
        "own experiment identity, then the same image is queried again. The legacy "
        "`top1_is_self` field reports final ranking; `retrieval_top1_is_self` and "
        "`final_top1_is_self` make the retrieval/final distinction explicit."
    )
    lines.append("")
    lines.append(
        "`rerank_policy=none` means retrieval-vector-only ranking. `rerank_policy=top1` "
        "means only the retrieval top-1 candidate is pairwise-reranked. "
        "`rerank_policy=full` is the old expensive behavior that reranks the entire "
        "shortlist."
    )
    lines.append("")
    lines.append(
        "For the advisor-facing 1000-sample experiment, `top1` or `none` is recommended "
        "because full Harris rerank is very expensive."
    )
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- Command: `{command}`")
    lines.append(f"- Table prefix: `{table_prefix}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append(f"- Git commit: `{git_info.get('commit', 'unknown')}`")
    lines.append(f"- Git dirty: `{str(bool(git_info.get('dirty'))).lower()}`")
    lines.append("")
    lines.append("## Method Results")
    lines.append("")
    lines.append(
        "| method | rerank policy | n_queries | retrieval top1 self-match | final top1 self-match | "
        "self in shortlist | mean retrieval rank | mean final rank | exact vector score | "
        "mean total ms | p95 total ms | reranked candidates | skipped rerank candidates |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {method} | {policy} | {n_queries} | {retrieval_top1} | {final_top1} | "
            "{shortlist} | {mean_retrieval_rank} | {mean_final_rank} | {exact} | "
            "{mean_ms} | {p95_ms} | {reranked_count} | {skipped_count} |".format(
                method=row.get("method", ""),
                policy=row.get("rerank_policy", ""),
                n_queries=row.get("n_queries", 0),
                retrieval_top1=_format_rate(row.get("retrieval_top1_self_match_rate")),
                final_top1=_format_rate(row.get("final_top1_self_match_rate")),
                shortlist=_format_rate(row.get("self_in_shortlist_rate")),
                mean_retrieval_rank=_format_number(row.get("mean_retrieval_rank_self"), digits=2),
                mean_final_rank=_format_number(row.get("mean_final_rank_self"), digits=2),
                exact=_format_rate(row.get("exact_self_vector_score_rate_epsilon")),
                mean_ms=_format_number(row.get("mean_total_query_ms"), digits=1),
                p95_ms=_format_number(row.get("p95_total_query_ms"), digits=1),
                reranked_count=_format_number(row.get("reranked_candidate_count_mean"), digits=1),
                skipped_count=_format_number(row.get("skipped_rerank_candidate_count_mean"), digits=1),
            )
        )
    lines.append("")

    warnings = []
    for row in summary_rows:
        policy = str(row.get("rerank_policy") or "").lower()
        rerank_rate = _as_float(row.get("rerank_performed_rate")) or 0.0
        source_rate = _as_float(row.get("candidate_source_available_rate")) or 0.0
        query_errors = int(_as_float(row.get("query_error_count")) or 0)
        if policy != "none" and rerank_rate < 1.0:
            warnings.append(f"- `{row.get('method')}` rerank was not performed for every successful query.")
        if policy == "full" and source_rate < 1.0:
            warnings.append(f"- `{row.get('method')}` had at least one unavailable candidate source.")
        if query_errors:
            warnings.append(f"- `{row.get('method')}` had {query_errors} query errors.")
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        lines.extend(warnings)
        lines.append("")

    lines.append("## DB Layout")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dict(db_layout), indent=2, sort_keys=True, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Scores are matcher-dependent; the script does not assume classical methods must return "
        "an exact score of 1.0 on self-pairs."
    )
    lines.append(
        "- `exact_self_vector_score_rate_epsilon` only checks retrieval scores that are numerically "
        "within the configured epsilon of 1.0."
    )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 1:N fingerprint identification self-match repeatability experiment "
            "using an isolated table prefix."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest CSV path. Defaults to data/manifests/<dataset>/manifest.csv.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--capture-filter", default=None)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated retrieval/rerank methods. Defaults to "
            f"{','.join(DEFAULT_METHODS)}. Use --include-vit to append vit."
        ),
    )
    parser.add_argument("--include-vit", action="store_true", help="Append vit to the advisor-facing method set.")
    parser.add_argument("--shortlist-size", type=int, default=DEFAULT_SHORTLIST_SIZE)
    parser.add_argument(
        "--rerank-policy",
        choices=RERANK_POLICIES,
        default=DEFAULT_RERANK_POLICY,
        help=(
            "Rerank policy for the identification shortlist: full reranks the full shortlist, "
            "top1 reranks only the retrieval top-1 candidate, and none keeps vector-only ranking."
        ),
    )
    parser.add_argument("--score-epsilon", type=float, default=DEFAULT_SCORE_EPSILON)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--table-prefix", default=None)
    parser.add_argument(
        "--reset-prefix",
        action="store_true",
        help="Drop only tables that use the non-empty experiment table prefix before running.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--identity-database-url", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _resolve_output_dir(raw: str | None, timestamp: str) -> Path:
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return (
        REPO_ROOT
        / "artifacts"
        / "reports"
        / "identification"
        / "self_match_repeatability"
        / timestamp
    )


def run_experiment(args: argparse.Namespace) -> int:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_prefix = normalize_table_prefix(args.table_prefix) if args.table_prefix else default_table_prefix(run_timestamp)
    if not table_prefix:
        raise ValueError("This experiment requires a non-empty table prefix.")

    methods = resolve_experiment_methods(args.methods, include_vit=bool(args.include_vit))
    rerank_policy = str(getattr(args, "rerank_policy", DEFAULT_RERANK_POLICY)).strip().lower()
    identify_kwargs_for_rerank_policy(rerank_policy)
    output_dir = _resolve_output_dir(args.output_dir, run_timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _format_command(sys.argv)

    print("=== Identification self-match repeatability experiment ===")
    print(f"Repo root    : {REPO_ROOT}")
    print(f"Dataset      : {args.dataset}")
    print(f"Table prefix : {table_prefix}")
    print(f"Methods      : {','.join(methods)}")
    print(f"Rerank policy: {rerank_policy}")
    print(f"Output dir   : {output_dir}")

    dropped_tables: list[str] = []
    if args.reset_prefix:
        print(f"[db] Resetting prefixed experiment tables for {table_prefix!r}")
        dropped_tables = reset_prefixed_experiment_tables(
            table_prefix=table_prefix,
            database_url=args.database_url,
            identity_database_url=args.identity_database_url,
        )
        print(f"[db] Dropped {len(dropped_tables)} prefixed table references.")

    resolver = CandidateSourceResolver()
    service = IdentificationService(
        database_url=args.database_url,
        identity_database_url=args.identity_database_url,
        table_prefix=table_prefix,
        candidate_source_resolver=resolver,
    )
    existing_count = service.store.total_people()
    if existing_count > 0 and not args.reset_prefix:
        raise RuntimeError(
            f"Table prefix {table_prefix!r} already contains {existing_count} enrolled identities. "
            "Use --reset-prefix for this experiment prefix or choose a new --table-prefix."
        )
    db_layout = service.store.dump_layout()

    selected_rows, selection_report = load_manifest_selection(
        dataset=args.dataset,
        manifest_path=args.manifest,
        repo_root=REPO_ROOT,
        limit=args.limit,
        seed=args.seed,
        capture_filter=args.capture_filter,
    )
    print(
        "[manifest] selected "
        f"{len(selected_rows)} rows from {selection_report.valid_row_count} valid rows "
        f"({selection_report.missing_path_count} missing image paths skipped)."
    )

    enrollment_rows, enrolled_rows = enroll_selected_rows(
        service=service,
        resolver=resolver,
        rows=selected_rows,
        methods=methods,
        fail_fast=bool(args.fail_fast),
    )
    write_csv(output_dir / "enrollment.csv", enrollment_rows, ENROLLMENT_COLUMNS)
    enroll_error_count = sum(1 for row in enrollment_rows if not _as_bool(row.get("enrolled")))
    print(f"[enroll] enrolled={len(enrolled_rows)} errors={enroll_error_count}")

    summary_rows: list[dict[str, Any]] = []
    query_failure_counts: dict[str, int] = {}
    for method in methods:
        query_rows, failure_rows = query_self_matches_for_method(
            service=service,
            method=method,
            rows=enrolled_rows,
            shortlist_size=int(args.shortlist_size),
            fail_fast=bool(args.fail_fast),
            rerank_policy=rerank_policy,
        )
        write_csv(output_dir / f"queries_{method}.csv", query_rows, QUERY_COLUMNS)
        if failure_rows:
            write_csv(output_dir / f"failures_{method}.csv", failure_rows, FAILURE_COLUMNS)
        query_failure_counts[method] = len(failure_rows)
        summary_rows.append(
            summarize_method_results(
                dataset=args.dataset,
                table_prefix=table_prefix,
                method=method,
                n_selected=len(selected_rows),
                n_enrolled=len(enrolled_rows),
                enroll_error_count=enroll_error_count,
                query_rows=query_rows,
                query_error_count=len(failure_rows),
                rerank_policy=rerank_policy,
                score_epsilon=float(args.score_epsilon),
            )
        )

    write_csv(output_dir / "results_summary.csv", summary_rows, SUMMARY_COLUMNS)

    git_info = collect_git_info()
    markdown = render_summary_markdown(
        summary_rows=summary_rows,
        command=command,
        table_prefix=table_prefix,
        db_layout=db_layout,
        git_info=git_info,
        output_dir=output_dir,
    )
    (output_dir / "results_summary.md").write_text(markdown, encoding="utf-8")

    run_manifest = {
        "run_timestamp": run_timestamp,
        "command": command,
        "repo_root": str(REPO_ROOT),
        "dataset": args.dataset,
        "manifest": selection_report.to_json(),
        "limit": int(args.limit),
        "seed": int(args.seed),
        "capture_filter": args.capture_filter,
        "methods": methods,
        "primary_default_methods": list(DEFAULT_METHODS),
        "optional_methods": list(OPTIONAL_METHODS),
        "shortlist_size": int(args.shortlist_size),
        "rerank_policy": rerank_policy,
        "score_epsilon": float(args.score_epsilon),
        "table_prefix": table_prefix,
        "reset_prefix": bool(args.reset_prefix),
        "dropped_tables": dropped_tables,
        "database_layout": db_layout,
        "n_selected": len(selected_rows),
        "n_enrolled": len(enrolled_rows),
        "enroll_error_count": enroll_error_count,
        "query_error_counts": query_failure_counts,
        "expected_mappings": [row.expected_mapping() for row in selected_rows],
        "git": git_info,
        "outputs": {
            "run_manifest": str(output_dir / "run_manifest.json"),
            "enrollment": str(output_dir / "enrollment.csv"),
            "results_summary_csv": str(output_dir / "results_summary.csv"),
            "results_summary_md": str(output_dir / "results_summary.md"),
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    print(f"[done] wrote {output_dir / 'results_summary.md'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
