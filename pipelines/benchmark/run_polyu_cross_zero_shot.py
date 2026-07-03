"""Thin PolyU Cross zero-shot benchmark runner (Phase 3A plumbing).

Runs *existing* benchmark scoring methods on the *existing* PolyU Cross pair
bundle and writes canonical score CSVs plus run metadata. It deliberately does
**not** train, fine-tune, calibrate, choose thresholds, regenerate pairs, edit
the manifest/pairs, or touch UI/API code.

Design
------
* Classical local-feature methods (``sift`` and friends) are scored by reusing
  the canonical ``pipelines/benchmark/evaluate.py`` orchestrator via a
  subprocess, exactly like ``run_benchmark_matrix.py`` does. The only new work
  here is (a) resolving PolyU Cross relative image paths against the configured
  dataset root before scoring and (b) enriching / auditing the produced scores.
* ``sourceafis_open`` reuses the existing SourceAFIS scoring internals
  (``score_pairs`` / ``TemplateCache`` / ``ensure_sourceafis_available`` from
  ``scripts/diagnostics/run_sourceafis_plain_roll_benchmark.py``). When the
  SourceAFIS sidecar is not configured the runner degrades gracefully and
  records per-pair failures instead of crashing.

Outputs (under ``--outdir``)
----------------------------
* ``scores_polyu_cross_<method>_<split>.csv`` - enriched canonical scores.
* ``run_<method>_<split>.meta.json``          - per-method run metadata.
* ``failures_polyu_cross_<method>_<split>.csv`` - unreadable/unscorable pairs.
* ``latency_summary.csv``                      - per method/split latency.
* ``run_manifest.json``                        - whole-run provenance manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fpbench.datasets.polyu_cross_pairs import (
    DATASET as POLYU_CROSS_DATASET,
    PolyUCrossPairError,
    balanced_limit,
    finger_position_series,
    iter_pair_split_csvs,
    load_polyu_cross_pairs,
    resolve_polyu_cross_root,
    resolve_pairs_frame,
)

try:
    from importlib.metadata import version as pkg_version  # type: ignore
except Exception:  # pragma: no cover - defensive
    pkg_version = None  # type: ignore

DATASET_NAME = POLYU_CROSS_DATASET
DEFAULT_MANIFEST_DIR = "data/manifests/polyu_cross"
DEFAULT_OUTDIR = "artifacts/reports/benchmark/polyu_cross_zero_shot_v0"
EVALUATE_PY = REPO_ROOT / "pipelines" / "benchmark" / "evaluate.py"

RUN_SCHEMA_VERSION = "polyu_cross_zero_shot_v0"

# Methods scored through the canonical classical evaluate.py path.
CLASSICAL_METHODS = ("sift", "sift_plain_roll_v2", "harris", "classic_v2", "minutiae")
# Methods scored through the SourceAFIS engine internals.
SOURCEAFIS_METHODS = ("sourceafis_open",)
SUPPORTED_METHODS = tuple(CLASSICAL_METHODS) + tuple(SOURCEAFIS_METHODS)
DEFAULT_METHODS = "sift"

# Canonical leading columns required of every score CSV. Method-specific
# diagnostic columns are appended after these.
REQUIRED_SCORE_COLUMNS = [
    "pair_id",
    "label",
    "split",
    "subject_a",
    "subject_b",
    "path_a",
    "path_b",
    "score",
    "method",
    "dataset",
]

IDENTITY_EXTRA_COLUMNS = [
    "finger_unit_a",
    "finger_unit_b",
    "frgp",
    "modality_a",
    "modality_b",
    "session_a",
    "session_b",
    "sample_uid_a",
    "sample_uid_b",
]

RESOLUTION_COLUMNS = [
    "resolved_path_a",
    "resolved_path_b",
    "path_a_exists",
    "path_b_exists",
    "status",
    "error",
]

FAILURE_COLUMNS = [
    "method",
    "dataset",
    "split",
    "pair_id",
    "label",
    "subject_a",
    "subject_b",
    "path_a",
    "path_b",
    "resolved_path_a",
    "resolved_path_b",
    "operation",
    "error_type",
    "error_message",
]

LATENCY_COLUMNS = [
    "method",
    "split",
    "n_pairs",
    "n_scored",
    "n_failed",
    "avg_ms_pair",
    "p50_ms_pair",
    "p95_ms_pair",
    "total_ms",
    "source",
]


class PolyUCrossRunError(RuntimeError):
    """Raised for unrecoverable runner setup/protocol failures."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def sha256_file(path: Path) -> Optional[str]:
    import hashlib

    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def safe_pkg_version(name: str) -> Optional[str]:
    if pkg_version is None:
        return None
    try:
        return pkg_version(name)
    except Exception:
        return None


def git_info() -> dict[str, Any]:
    def _run(args: list[str]) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True
            )
            if proc.returncode != 0:
                return None
            return proc.stdout.strip()
        except Exception:
            return None

    commit = _run(["rev-parse", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": None if status is None else bool(status.strip()),
    }


def _finite_float_array(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def method_family(method: str) -> str:
    if method in SOURCEAFIS_METHODS:
        return "sourceafis"
    if method in CLASSICAL_METHODS:
        return "classical"
    raise PolyUCrossRunError(
        f"Unsupported method {method!r}. Supported: {list(SUPPORTED_METHODS)}"
    )


# ---------------------------------------------------------------------------
# Enriched score-frame assembly (shared)
# ---------------------------------------------------------------------------
def build_enriched_scores(
    resolved_df: pd.DataFrame,
    *,
    method: str,
    split: str,
    scores: np.ndarray,
    status: list[str],
    errors: list[str],
    method_extra: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Assemble a canonical enriched score frame preserving pair identity."""

    n = len(resolved_df)
    frame: dict[str, Any] = {
        "pair_id": resolved_df["pair_id"].values,
        "label": resolved_df["label"].astype(int).values,
        "split": resolved_df["split"].astype(str).values if "split" in resolved_df else [split] * n,
        "subject_a": resolved_df["subject_a"].values,
        "subject_b": resolved_df["subject_b"].values,
        "path_a": resolved_df["path_a"].astype(str).values,
        "path_b": resolved_df["path_b"].astype(str).values,
        "score": np.asarray(scores, dtype=float),
        "method": [method] * n,
        "dataset": [DATASET_NAME] * n,
    }
    for column in IDENTITY_EXTRA_COLUMNS:
        if column in resolved_df.columns:
            frame[column] = resolved_df[column].values
    for column in ("resolved_path_a", "resolved_path_b", "path_a_exists", "path_b_exists"):
        if column in resolved_df.columns:
            frame[column] = resolved_df[column].values
    frame["status"] = status
    frame["error"] = errors
    if method_extra:
        for column, values in method_extra.items():
            frame[column] = values

    df = pd.DataFrame(frame)

    leading = [c for c in REQUIRED_SCORE_COLUMNS if c in df.columns]
    identity = [c for c in IDENTITY_EXTRA_COLUMNS if c in df.columns]
    resolution = [c for c in RESOLUTION_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in (leading + identity + resolution)]
    return df[leading + identity + resolution + extras]


def failures_from_scores(scored: pd.DataFrame, *, method: str, split: str) -> pd.DataFrame:
    """Extract a failure frame from enriched scores where status != ok."""

    failed = scored[scored["status"].astype(str) != "ok"]
    rows: list[dict[str, Any]] = []
    for _, row in failed.iterrows():
        rows.append(
            {
                "method": method,
                "dataset": DATASET_NAME,
                "split": split,
                "pair_id": row.get("pair_id", ""),
                "label": row.get("label", ""),
                "subject_a": row.get("subject_a", ""),
                "subject_b": row.get("subject_b", ""),
                "path_a": row.get("path_a", ""),
                "path_b": row.get("path_b", ""),
                "resolved_path_a": row.get("resolved_path_a", ""),
                "resolved_path_b": row.get("resolved_path_b", ""),
                "operation": "score_pair",
                "error_type": "unscorable_pair",
                "error_message": str(row.get("error", "")),
            }
        )
    return pd.DataFrame(rows, columns=FAILURE_COLUMNS)


# ---------------------------------------------------------------------------
# Classical scoring via evaluate.py
# ---------------------------------------------------------------------------
def _run_evaluate_subprocess(
    *,
    method: str,
    split: str,
    manifest_dir: Path,
    resolved_pairs_csv: Path,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Invoke the canonical evaluate.py on a resolved pairs CSV.

    Returns ``(raw_scores_csv, eval_meta_json)``. Raises on non-zero exit.
    """

    raw_scores = work_dir / f"raw_scores_{method}_{split}.csv"
    roc_png = work_dir / f"roc_{method}_{split}.png"
    eval_run_meta = work_dir / f"eval_run_{method}_{split}.meta.json"
    summary_csv = work_dir / "eval_results_summary.csv"

    cmd = [
        sys.executable,
        str(EVALUATE_PY),
        "--method",
        method,
        "--split",
        split,
        "--limit",
        "0",  # limiting is applied by this runner before writing resolved pairs
        "--dataset",
        DATASET_NAME,
        "--data_dir",
        str(manifest_dir),
        "--pairs_file",
        str(resolved_pairs_csv),
        "--pair_set_name",
        split,
        "--out_scores",
        str(raw_scores),
        "--out_roc",
        str(roc_png),
        "--out_run_meta",
        str(eval_run_meta),
        "--summary_csv",
        str(summary_csv),
    ]

    env = os.environ.copy()
    env["FPRJ_ROOT"] = str(REPO_ROOT)
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("MPLBACKEND", "Agg")  # headless-safe ROC rendering

    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
        raise PolyUCrossRunError(
            f"evaluate.py failed for method={method} split={split} (exit {proc.returncode}):\n"
            + "\n".join(tail)
        )
    if not raw_scores.exists():
        raise PolyUCrossRunError(
            f"evaluate.py did not produce scores for method={method} split={split}: {raw_scores}"
        )
    return raw_scores, eval_run_meta


def score_split_classical(
    *,
    method: str,
    split: str,
    manifest_dir: Path,
    resolved_df: pd.DataFrame,
    work_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one split with a classical method and enrich the output.

    Returns ``(enriched_scores_df, timing_meta)``.
    """

    ensure_dir(work_dir)
    # eval_classic reads path_a/path_b literally; feed it resolved absolute paths
    # while preserving the identity columns for our own post-join audit.
    tmp_pairs = pd.DataFrame(
        {
            "pair_id": resolved_df["pair_id"].values,
            "label": resolved_df["label"].astype(int).values,
            "split": resolved_df["split"].astype(str).values,
            "subject_a": resolved_df["subject_a"].values,
            "subject_b": resolved_df["subject_b"].values,
            "frgp": resolved_df["frgp"].values if "frgp" in resolved_df.columns else [0] * len(resolved_df),
            "path_a": resolved_df["resolved_path_a"].astype(str).values,
            "path_b": resolved_df["resolved_path_b"].astype(str).values,
        }
    )
    resolved_pairs_csv = work_dir / f"resolved_pairs_{method}_{split}.csv"
    tmp_pairs.to_csv(resolved_pairs_csv, index=False)

    raw_scores_csv, eval_meta_json = _run_evaluate_subprocess(
        method=method,
        split=split,
        manifest_dir=manifest_dir,
        resolved_pairs_csv=resolved_pairs_csv,
        work_dir=work_dir,
    )

    raw = pd.read_csv(raw_scores_csv).reset_index(drop=True)
    base = resolved_df.reset_index(drop=True)
    if len(raw) != len(base):
        raise PolyUCrossRunError(
            f"Score row count {len(raw)} does not match pair count {len(base)} "
            f"for method={method} split={split}; refusing to misalign pair_id."
        )
    # Positional join is valid because evaluate.py preserves input pair order
    # with --limit 0. Verify labels line up before trusting the alignment.
    if not np.array_equal(base["label"].astype(int).values, raw["label"].astype(int).values):
        raise PolyUCrossRunError(
            f"Label order mismatch between resolved pairs and scores for "
            f"method={method} split={split}; refusing to misalign pair_id/label."
        )

    scores = pd.to_numeric(raw["score"], errors="coerce").to_numpy(dtype=float)
    k1 = pd.to_numeric(raw.get("k1", pd.Series([0] * len(raw))), errors="coerce").fillna(0).astype(int)
    k2 = pd.to_numeric(raw.get("k2", pd.Series([0] * len(raw))), errors="coerce").fillna(0).astype(int)

    exists_a = base.get("path_a_exists", pd.Series([True] * len(base))).astype(bool).values
    exists_b = base.get("path_b_exists", pd.Series([True] * len(base))).astype(bool).values

    status: list[str] = []
    errors: list[str] = []
    for i in range(len(base)):
        if not exists_a[i] or not exists_b[i]:
            missing = []
            if not exists_a[i]:
                missing.append("path_a")
            if not exists_b[i]:
                missing.append("path_b")
            status.append("failed")
            errors.append(f"missing_image:{'+'.join(missing)}")
        elif int(k1.iloc[i]) == 0 or int(k2.iloc[i]) == 0:
            status.append("failed")
            errors.append("no_features_extracted")
        elif not np.isfinite(scores[i]):
            status.append("failed")
            errors.append("non_finite_score")
        else:
            status.append("ok")
            errors.append("")

    method_extra = {
        column: raw[column].values
        for column in ("inliers", "matches", "k1", "k2", "extract_a_ms", "extract_b_ms", "match_ms", "pair_total_ms")
        if column in raw.columns
    }
    enriched = build_enriched_scores(
        base,
        method=method,
        split=split,
        scores=scores,
        status=status,
        errors=errors,
        method_extra=method_extra,
    )

    # eval_classic reports per-pair latency in the score rows (pair_total_ms);
    # summarize it directly for the latency summary.
    values = _finite_float_array(raw.get("pair_total_ms", pd.Series(dtype=float)))
    timing = {
        "avg_ms_pair": float(np.mean(values)) if values.size else float("nan"),
        "p50_ms_pair": float(np.median(values)) if values.size else float("nan"),
        "p95_ms_pair": float(np.quantile(values, 0.95)) if values.size else float("nan"),
        "total_ms": float(np.sum(values)) if values.size else float("nan"),
        "source": "classic_pair_total_ms",
    }
    return enriched, timing


# ---------------------------------------------------------------------------
# SourceAFIS scoring via existing engine internals
# ---------------------------------------------------------------------------
def _all_failures(
    resolved_df: pd.DataFrame,
    *,
    method: str,
    split: str,
    error_type: str,
    error_message: str,
    method_extra: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(resolved_df)
    scores = np.full(n, np.nan, dtype=float)
    status = ["failed"] * n
    errors = [error_message] * n
    enriched = build_enriched_scores(
        resolved_df,
        method=method,
        split=split,
        scores=scores,
        status=status,
        errors=errors,
        method_extra=method_extra,
    )
    enriched["error_type"] = error_type
    timing = {
        "avg_ms_pair": float("nan"),
        "p50_ms_pair": float("nan"),
        "p95_ms_pair": float("nan"),
        "total_ms": float("nan"),
        "source": "all_failed",
    }
    return enriched, timing


def score_split_sourceafis(
    *,
    split: str,
    resolved_df: pd.DataFrame,
    work_dir: Path,
    dpi_strategy: str = "default",
    image_dpi: Optional[int] = None,
    engine: Any = None,
    require_enabled_env: Optional[bool] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one split with SourceAFIS, degrading gracefully when unavailable.

    ``engine`` may be injected (tests); otherwise the real provider is resolved
    and requires the SourceAFIS sidecar environment.
    """

    method = "sourceafis_open"
    ensure_dir(work_dir)

    try:
        import scripts.diagnostics.run_sourceafis_plain_roll_benchmark as sa
    except Exception as exc:  # pragma: no cover - import environment dependent
        return _all_failures(
            resolved_df,
            method=method,
            split=split,
            error_type=type(exc).__name__,
            error_message=f"sourceafis_module_import_failed: {exc}",
        )

    if require_enabled_env is None:
        require_enabled_env = engine is None

    try:
        selected_engine, metadata = sa.ensure_sourceafis_available(
            engine=engine, require_enabled_env=require_enabled_env
        )
    except Exception as exc:
        return _all_failures(
            resolved_df,
            method=method,
            split=split,
            error_type=type(exc).__name__,
            error_message=f"sourceafis_unavailable: {exc}",
        )

    pairs = pd.DataFrame(
        {
            "dataset": [DATASET_NAME] * len(resolved_df),
            "split": resolved_df["split"].astype(str).values,
            "pair_id": resolved_df["pair_id"].astype(str).values,
            "label": resolved_df["label"].astype(int).values,
            "subject_a": resolved_df["subject_a"].astype(str).values,
            "subject_b": resolved_df["subject_b"].astype(str).values,
            "finger_position": finger_position_series(resolved_df).values,
            "path_a": resolved_df["resolved_path_a"].astype(str).values,
            "path_b": resolved_df["resolved_path_b"].astype(str).values,
        }
    )

    try:
        service_url = str(os.getenv(sa.SOURCEAFIS_SERVICE_URL_ENV, "") or metadata.metadata.get("service_url", ""))
        cache = sa.TemplateCache(
            work_dir / "sourceafis_template_cache",
            provider_metadata=metadata,
            service_url=service_url,
            dpi_strategy=dpi_strategy,
            image_dpi=image_dpi,
            repo_root=REPO_ROOT,
        )
        scores_df, _failures, _latency = sa.score_pairs(
            pairs,
            engine=selected_engine,
            cache=cache,
            retry_config=sa.RetryConfig(),
        )
    except Exception as exc:
        return _all_failures(
            resolved_df,
            method=method,
            split=split,
            error_type=type(exc).__name__,
            error_message=f"sourceafis_scoring_failed: {exc}",
        )

    scores_df = scores_df.reset_index(drop=True)
    base = resolved_df.reset_index(drop=True)
    # Align by pair_id to be robust to any internal reordering.
    order = {str(pid): i for i, pid in enumerate(base["pair_id"].astype(str).values)}
    scores_df["_order"] = scores_df["pair_id"].astype(str).map(order)
    scores_df = scores_df.sort_values("_order", kind="mergesort").reset_index(drop=True)

    raw_scores = pd.to_numeric(scores_df.get("raw_score", pd.Series([np.nan] * len(base))), errors="coerce")
    scores = raw_scores.to_numpy(dtype=float)
    errors_col = scores_df.get("error", pd.Series([""] * len(base))).astype(str)

    status: list[str] = []
    errors: list[str] = []
    for i in range(len(base)):
        err = errors_col.iloc[i] if i < len(errors_col) else ""
        if err and err.strip():
            status.append("failed")
            errors.append(err)
        elif not np.isfinite(scores[i]):
            status.append("failed")
            errors.append("non_finite_score")
        else:
            status.append("ok")
            errors.append("")

    extra_columns = [
        "raw_score",
        "score_semantics",
        "higher_is_more_similar",
        "provider_id",
        "provider_version",
        "extraction_latency_ms_a",
        "extraction_latency_ms_b",
        "verification_latency_ms",
        "verification_wall_latency_ms",
    ]
    method_extra = {
        column: scores_df[column].values for column in extra_columns if column in scores_df.columns
    }
    pair_total = (
        pd.to_numeric(scores_df.get("extraction_latency_ms_a", 0), errors="coerce").fillna(0)
        + pd.to_numeric(scores_df.get("extraction_latency_ms_b", 0), errors="coerce").fillna(0)
        + pd.to_numeric(scores_df.get("verification_latency_ms", 0), errors="coerce").fillna(0)
    )
    method_extra["pair_total_ms"] = pair_total.values

    enriched = build_enriched_scores(
        base,
        method=method,
        split=split,
        scores=scores,
        status=status,
        errors=errors,
        method_extra=method_extra,
    )

    values = _finite_float_array(pair_total)
    timing = {
        "avg_ms_pair": float(np.mean(values)) if values.size else float("nan"),
        "p50_ms_pair": float(np.median(values)) if values.size else float("nan"),
        "p95_ms_pair": float(np.quantile(values, 0.95)) if values.size else float("nan"),
        "total_ms": float(np.sum(values)) if values.size else float("nan"),
        "source": "sourceafis_pair_total_ms",
    }
    return enriched, timing


# ---------------------------------------------------------------------------
# Per method/split driver
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    method: str
    split: str
    pairs_csv: Path
    scores_csv: Path
    run_meta_json: Path
    failures_csv: Path
    n_pairs: int
    n_scored: int
    n_failed: int
    n_positive: int
    n_negative: int
    timing: dict[str, Any]


def run_method_split(
    *,
    method: str,
    split: str,
    pairs_csv: Path,
    manifest_dir: Path,
    root: Optional[Path],
    outdir: Path,
    work_dir: Path,
    limit: int,
    strict: bool,
    dpi_strategy: str,
    image_dpi: Optional[int],
    sourceafis_engine: Any = None,
) -> SplitResult:
    pairs = load_polyu_cross_pairs(pairs_csv)
    if int(limit) > 0:
        pairs = balanced_limit(pairs, int(limit))
    resolved = resolve_pairs_frame(pairs, root)

    if strict:
        missing = resolved[(~resolved["path_a_exists"]) | (~resolved["path_b_exists"])]
        if not missing.empty:
            raise PolyUCrossRunError(
                f"[strict] {len(missing)} pair(s) reference missing images for "
                f"method={method} split={split}. First: "
                f"{missing.iloc[0]['resolved_path_a']!r} / {missing.iloc[0]['resolved_path_b']!r}"
            )

    family = method_family(method)
    if family == "classical":
        try:
            enriched, timing = score_split_classical(
                method=method,
                split=split,
                manifest_dir=manifest_dir,
                resolved_df=resolved,
                work_dir=work_dir,
            )
        except PolyUCrossRunError:
            if strict:
                raise
            enriched, timing = _all_failures(
                resolved,
                method=method,
                split=split,
                error_type="classical_scoring_failed",
                error_message="classical_scoring_failed",
            )
    else:
        enriched, timing = score_split_sourceafis(
            split=split,
            resolved_df=resolved,
            work_dir=work_dir,
            dpi_strategy=dpi_strategy,
            image_dpi=image_dpi,
            engine=sourceafis_engine,
        )

    scores_csv = outdir / f"scores_{DATASET_NAME}_{method}_{split}.csv"
    enriched.to_csv(scores_csv, index=False)

    failures = failures_from_scores(enriched, method=method, split=split)
    failures_csv = outdir / f"failures_{DATASET_NAME}_{method}_{split}.csv"
    failures.to_csv(failures_csv, index=False)

    n_pairs = int(len(enriched))
    n_failed = int((enriched["status"].astype(str) != "ok").sum())
    n_scored = n_pairs - n_failed
    n_positive = int((enriched["label"].astype(int) == 1).sum())
    n_negative = int((enriched["label"].astype(int) == 0).sum())

    run_meta = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "method": method,
        "method_family": family,
        "dataset": DATASET_NAME,
        "split": split,
        "pairs_csv": str(pairs_csv),
        "pairs_csv_sha256": sha256_file(pairs_csv),
        "scores_csv": str(scores_csv),
        "failures_csv": str(failures_csv),
        "limit": int(limit),
        "strict": bool(strict),
        "n_pairs": n_pairs,
        "n_scored": n_scored,
        "n_failed": n_failed,
        "label_counts": {"n_positive": n_positive, "n_negative": n_negative},
        "timing": timing,
        "config": {
            "resolves_relative_paths": True,
            "regenerates_pairs": False,
            "trains_or_calibrates": False,
            "score_columns": list(enriched.columns),
            "dpi_strategy": dpi_strategy if family == "sourceafis" else None,
        },
    }
    run_meta_json = outdir / f"run_{DATASET_NAME}_{method}_{split}.meta.json"
    run_meta_json.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return SplitResult(
        method=method,
        split=split,
        pairs_csv=pairs_csv,
        scores_csv=scores_csv,
        run_meta_json=run_meta_json,
        failures_csv=failures_csv,
        n_pairs=n_pairs,
        n_scored=n_scored,
        n_failed=n_failed,
        n_positive=n_positive,
        n_negative=n_negative,
        timing=timing,
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def run(
    *,
    manifest_dir: Path,
    outdir: Path,
    methods: list[str],
    splits: list[str],
    limit: int,
    strict: bool,
    polyu_root: Optional[str],
    dpi_strategy: str,
    image_dpi: Optional[int],
    sourceafis_engine: Any = None,
) -> dict[str, Any]:
    manifest_dir = Path(manifest_dir)
    if not (manifest_dir / "manifest.csv").exists():
        raise PolyUCrossRunError(f"PolyU Cross manifest.csv not found under {manifest_dir}")

    unsupported = [m for m in methods if m not in SUPPORTED_METHODS]
    if unsupported:
        raise PolyUCrossRunError(
            f"Unsupported method(s): {unsupported}. Supported: {list(SUPPORTED_METHODS)}"
        )

    ensure_dir(outdir)
    work_dir = outdir / "_work"
    ensure_dir(work_dir)

    resolved_root = resolve_polyu_cross_root(manifest_dir, override=polyu_root)

    split_csvs = dict(iter_pair_split_csvs(manifest_dir, splits))

    results: list[SplitResult] = []
    for method in methods:
        for split in splits:
            pairs_csv = split_csvs[split]
            if not pairs_csv.exists():
                raise PolyUCrossRunError(f"Missing PolyU Cross pairs CSV for split={split}: {pairs_csv}")
            print(f"[RUN] method={method} split={split} pairs={pairs_csv.name}")
            result = run_method_split(
                method=method,
                split=split,
                pairs_csv=pairs_csv,
                manifest_dir=manifest_dir,
                root=resolved_root.root,
                outdir=outdir,
                work_dir=work_dir,
                limit=limit,
                strict=strict,
                dpi_strategy=dpi_strategy,
                image_dpi=image_dpi,
                sourceafis_engine=sourceafis_engine,
            )
            results.append(result)
            print(
                f"[DONE] method={method} split={split} "
                f"scored={result.n_scored} failed={result.n_failed} -> {result.scores_csv.name}"
            )

    # Latency summary
    latency_rows = [
        {
            "method": r.method,
            "split": r.split,
            "n_pairs": r.n_pairs,
            "n_scored": r.n_scored,
            "n_failed": r.n_failed,
            "avg_ms_pair": r.timing.get("avg_ms_pair"),
            "p50_ms_pair": r.timing.get("p50_ms_pair"),
            "p95_ms_pair": r.timing.get("p95_ms_pair"),
            "total_ms": r.timing.get("total_ms"),
            "source": r.timing.get("source"),
        }
        for r in results
    ]
    latency_csv = outdir / "latency_summary.csv"
    pd.DataFrame(latency_rows, columns=LATENCY_COLUMNS).to_csv(latency_csv, index=False)

    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": DATASET_NAME,
        "repo_root": str(REPO_ROOT),
        "manifest_dir": str(manifest_dir),
        "outdir": str(outdir),
        "methods": list(methods),
        "splits": list(splits),
        "limit": int(limit),
        "strict": bool(strict),
        "polyu_root": {
            "path": str(resolved_root.root) if resolved_root.root is not None else None,
            "source": resolved_root.source,
            "exists": bool(resolved_root.exists),
        },
        "runs": [
            {
                "method": r.method,
                "split": r.split,
                "pairs_csv": str(r.pairs_csv),
                "pairs_csv_sha256": sha256_file(r.pairs_csv),
                "scores_csv": str(r.scores_csv),
                "run_meta_json": str(r.run_meta_json),
                "failures_csv": str(r.failures_csv),
                "n_pairs": r.n_pairs,
                "n_scored": r.n_scored,
                "n_failed": r.n_failed,
                "n_positive": r.n_positive,
                "n_negative": r.n_negative,
            }
            for r in results
        ],
        "latency_summary_csv": str(latency_csv),
        "git": git_info(),
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "packages": {
            "numpy": safe_pkg_version("numpy"),
            "opencv-python": safe_pkg_version("opencv-python"),
            "pandas": safe_pkg_version("pandas"),
            "scikit-learn": safe_pkg_version("scikit-learn"),
        },
        "constraints": {
            "regenerated_pairs": False,
            "modified_manifest_or_pairs": False,
            "trained_or_finetuned": False,
            "calibrated_or_chose_thresholds": False,
            "modified_ui_or_api": False,
            "copied_biometric_images": False,
        },
        "notes": [
            "Zero-shot / existing-method scoring only; no thresholds or final TAR/FAR claims.",
            "Relative image paths resolved against the configured PolyU dataset root.",
            "Pairs are read from the existing Phase 2A bundle and never regenerated.",
        ],
    }
    manifest_json = outdir / "run_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "manifest_json": manifest_json,
        "latency_csv": latency_csv,
        "results": results,
        "polyu_root": resolved_root,
    }


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run existing benchmark methods on the existing PolyU Cross pair bundle "
            "(zero-shot plumbing). Does not train, calibrate, choose thresholds, "
            "regenerate pairs, or modify the manifest/pairs."
        )
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_MANIFEST_DIR, help="PolyU Cross manifest dir.")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--methods",
        type=str,
        default=DEFAULT_METHODS,
        help=f"Comma-separated methods. Supported: {', '.join(SUPPORTED_METHODS)}.",
    )
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--limit", type=int, default=0, help="If >0, score only the first N balanced pairs (smoke).")
    parser.add_argument("--strict", action="store_true", help="Fail (do not degrade) when images are missing.")
    parser.add_argument(
        "--polyu_root",
        type=str,
        default="",
        help="Override the PolyU dataset root used to resolve relative image paths.",
    )
    parser.add_argument("--sourceafis_dpi_strategy", type=str, default="default", choices=["explicit", "infer_from_path", "default"])
    parser.add_argument("--sourceafis_image_dpi", type=int, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    manifest_dir = resolve_repo_path(args.data_dir)
    outdir = resolve_repo_path(args.outdir)
    methods = parse_csv_list(args.methods)
    splits = parse_csv_list(args.splits)
    if not methods:
        print("ERROR: --methods resolved to empty", file=sys.stderr)
        return 2
    if not splits:
        print("ERROR: --splits resolved to empty", file=sys.stderr)
        return 2

    try:
        summary = run(
            manifest_dir=manifest_dir,
            outdir=outdir,
            methods=methods,
            splits=splits,
            limit=int(args.limit),
            strict=bool(args.strict),
            polyu_root=str(args.polyu_root).strip() or None,
            dpi_strategy=str(args.sourceafis_dpi_strategy),
            image_dpi=args.sourceafis_image_dpi,
        )
    except (PolyUCrossRunError, PolyUCrossPairError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results: list[SplitResult] = summary["results"]
    total_scored = sum(r.n_scored for r in results)
    total_failed = sum(r.n_failed for r in results)
    print("\n=== PolyU Cross zero-shot run complete ===")
    print(f"Output dir     : {outdir}")
    print(f"Run manifest   : {summary['manifest_json']}")
    print(f"Latency summary: {summary['latency_csv']}")
    print(f"PolyU root     : {summary['polyu_root'].root} (source={summary['polyu_root'].source}, exists={summary['polyu_root'].exists})")
    print(f"Total scored   : {total_scored}")
    print(f"Total failed   : {total_failed}")
    for r in results:
        print(
            f"  - {r.method}/{r.split}: pairs={r.n_pairs} scored={r.n_scored} failed={r.n_failed} -> {r.scores_csv.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
