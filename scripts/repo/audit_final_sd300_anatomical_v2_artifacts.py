from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

for _optional_pandas_module in ("numexpr", "bottleneck"):
    sys.modules.setdefault(_optional_pandas_module, None)

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "final_sd300_anatomical_v2_artifact_audit"
)

DATASETS = ("nist_sd300b", "nist_sd300c")
SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {
    "train": {"pair_count": 28052, "positive_count": 7013, "negative_count": 21039},
    "val": {"pair_count": 3508, "positive_count": 877, "negative_count": 2631},
    "test": {"pair_count": 3556, "positive_count": 889, "negative_count": 2667},
}
EXPECTED_FRGP = tuple(range(1, 11))
OLD_8_FINGER_COUNT_TEXT = (
    "70" "29",
    "210" "87",
    "281" "16",
    "70" "3",
    "210" "9",
    "28" "12",
    "71" "1",
    "213" "3",
    "28" "44",
)
OLD_8_FINGER_COUNTS = {int(value) for value in OLD_8_FINGER_COUNT_TEXT}

CLASS_FINAL = "FINAL_SD300_ANATOMICAL_V2"
CLASS_COMPAT = "COMPATIBILITY_CURRENT_V2"
CLASS_LEGACY = "LEGACY_KNOWN_DO_NOT_USE"
CLASS_UNKNOWN = "UNKNOWN_NEEDS_REVIEW"

BENCHMARK_ROOT_REL = "artifacts/reports/benchmark"
FINAL_ALLOWLIST_DIRS = {
    "plain_roll_final_baselines_v2_anatomical_full_pairs",
    "plain_roll_final_sourceafis_v2_anatomical_full_pairs",
    "plain_roll_train_scores_v2_anatomical_full_pairs",
    "plain_roll_final_fusion_v1_v2_anatomical_full_pairs",
    "deep_pair_reranker_fast_ddp_anatomical_v2_ddp_scores",
    "fusion_v2_three_way_anatomical_v2_ddpdeep_comparison",
    "sourceafis_sift_quality_deep_fusion_v2_statistical_anatomical_v2_ddpdeep",
    "sourceafis_sift_quality_deep_group_weighted_fusion_v2_manual_45_15_30_10_anatomical_v2_ddpdeep",
    "sourceafis_sift_quality_deep_group_weighted_fusion_v2_auto_val_tar_at_far_001_anatomical_v2_ddpdeep",
    "kaggle_final_packages",
}
FINAL_NON_PAIR_ALLOWED_DIRS = {
    "fusion_v2_three_way_anatomical_v2_ddpdeep_comparison",
    "kaggle_final_packages",
}
COMPATIBILITY_CANDIDATE_DIRS = {
    "deep_pair_reranker_fast_ddp_full_pairs": {
        "required": {
            ("nist_sd300b", "val"),
            ("nist_sd300b", "test"),
            ("nist_sd300c", "val"),
            ("nist_sd300c", "test"),
        }
    },
    "deep_pair_reranker_fast_ddp_train_scores": {
        "required": {
            ("nist_sd300b", "train"),
            ("nist_sd300c", "train"),
        }
    },
}
KNOWN_LEGACY_DIRS = {
    "plain_roll_final_baselines_v1",
    "plain_roll_final_sourceafis_v1",
    "plain_roll_final_fusion_v1",
    "plain_roll_final_fusion_v1_full_pairs",
    "plain_roll_full_scores_v1",
    "plain_roll_fusion_ablation_v1",
    "plain_roll_train_scores_v1",
    "sourceafis_sift_deep_score_fusion_v2_proto_full_pairs",
    "sourceafis_sift_quality_deep_fusion_v2_debug_no_quality",
    "sourceafis_sift_quality_deep_fusion_v2_full_pairs",
    "sourceafis_sift_quality_deep_group_weighted_fusion_v2_auto_auc",
    "sourceafis_sift_quality_deep_group_weighted_fusion_v2_auto_tar_far_001",
    "sourceafis_sift_quality_deep_group_weighted_fusion_v2_manual_45_15_30_10",
}
LEGACY_REFERENCE_TOKENS = tuple(
    sorted(
        {
            *(f"{BENCHMARK_ROOT_REL}/{name}" for name in KNOWN_LEGACY_DIRS),
            *KNOWN_LEGACY_DIRS,
            "artifacts/checkpoints/deep_pair_reranker_fast_ddp",
            "artifacts/checkpoints/deep_pair_reranker_v1",
            "deep_pair_reranker_fast_ddp_train_scores",
        },
        key=len,
        reverse=True,
    )
)
OLD_TRAIN_SCORE_TOKENS = (
    "plain_roll_train_scores_v1",
    "deep_pair_reranker_fast_ddp_train_scores",
    "artifacts/checkpoints/deep_pair_reranker_fast_ddp/best.pt",
    "artifacts\\checkpoints\\deep_pair_reranker_fast_ddp\\best.pt",
)
FINAL_PREFINAL_SCAN_FILES = (
    "pipelines/benchmark/generate_plain_roll_train_scores_v2.py",
    "pipelines/benchmark/run_plain_roll_final_benchmark.py",
    "pipelines/benchmark/run_sourceafis_plain_roll_final_benchmark.py",
    "pipelines/benchmark/run_sourceafis_sift_quality_fusion_benchmark.py",
    "pipelines/benchmark/train_sourceafis_sift_quality_fusion.py",
    "pipelines/benchmark/train_run_sourceafis_sift_quality_deep_fusion_v2.py",
    "scripts/diagnostics/build_current_fusion_v2_diagnostics.py",
    "scripts/deep/score_fast_pair_ddp_splits.py",
    "scripts/diagnostics/analyze_true_accept_failures_across_methods.py",
    "scripts/diagnostics/build_deep_fusion_v2_failure_taxonomy.py",
    "scripts/diagnostics/prove_deep_fusion_v2_statistical_weight_replay.py",
    "src/fpbench/universal/deep_fusion_v2.py",
)
CSV_COLUMNS = (
    "artifact_path",
    "classification",
    "do_not_use",
    "is_final_allowed",
    "reason",
    "datasets_found",
    "splits_found",
    "row_count_status",
    "label_count_status",
    "frgp_status",
    "pair_sha_status",
    "legacy_signals",
    "unknown_signals",
    "recommended_action",
)
TEXT_SUFFIXES = {".json", ".md", ".py", ".txt", ".yaml", ".yml"}
MAX_TEXT_SCAN_BYTES = 2_000_000


@dataclass(frozen=True)
class CanonicalBundle:
    dataset: str
    split: str
    path: Path
    sha256: str
    pair_count: int
    positive_count: int
    negative_count: int
    frgp_coverage: tuple[int, ...]
    frame: pd.DataFrame


@dataclass
class TableEvidence:
    path: str
    dataset: str
    split: str
    rows: int
    positive_count: int | None
    negative_count: int | None
    frgp_coverage: tuple[int, ...]
    row_count_status: str
    label_count_status: str
    frgp_status: str
    pair_sha_status: str
    canonical_pair_sha256: str
    legacy_signals: list[str] = field(default_factory=list)
    unknown_signals: list[str] = field(default_factory=list)

    @property
    def passed_v2(self) -> bool:
        return (
            self.row_count_status.startswith("pass")
            and self.label_count_status.startswith("pass")
            and self.frgp_status.startswith("pass")
            and self.pair_sha_status.startswith("pass")
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "dataset": self.dataset,
            "split": self.split,
            "rows": self.rows,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "frgp_coverage": list(self.frgp_coverage),
            "row_count_status": self.row_count_status,
            "label_count_status": self.label_count_status,
            "frgp_status": self.frgp_status,
            "pair_sha_status": self.pair_sha_status,
            "canonical_pair_sha256": self.canonical_pair_sha256,
            "legacy_signals": list(self.legacy_signals),
            "unknown_signals": list(self.unknown_signals),
        }


@dataclass
class ArtifactAuditRow:
    artifact_path: str
    classification: str
    do_not_use: bool
    is_final_allowed: bool
    reason: str
    datasets_found: tuple[str, ...]
    splits_found: tuple[str, ...]
    row_count_status: str
    label_count_status: str
    frgp_status: str
    pair_sha_status: str
    legacy_signals: tuple[str, ...]
    unknown_signals: tuple[str, ...]
    recommended_action: str
    evidence: list[TableEvidence] = field(default_factory=list)

    def as_csv_row(self) -> dict[str, str]:
        return {
            "artifact_path": self.artifact_path,
            "classification": self.classification,
            "do_not_use": str(self.do_not_use).lower(),
            "is_final_allowed": str(self.is_final_allowed).lower(),
            "reason": self.reason,
            "datasets_found": _join(self.datasets_found),
            "splits_found": _join(self.splits_found),
            "row_count_status": self.row_count_status,
            "label_count_status": self.label_count_status,
            "frgp_status": self.frgp_status,
            "pair_sha_status": self.pair_sha_status,
            "legacy_signals": _join(self.legacy_signals),
            "unknown_signals": _join(self.unknown_signals),
            "recommended_action": self.recommended_action,
        }

    def as_json(self) -> dict[str, Any]:
        payload = self.as_csv_row()
        payload["do_not_use"] = self.do_not_use
        payload["is_final_allowed"] = self.is_final_allowed
        payload["datasets_found"] = list(self.datasets_found)
        payload["splits_found"] = list(self.splits_found)
        payload["legacy_signals"] = list(self.legacy_signals)
        payload["unknown_signals"] = list(self.unknown_signals)
        payload["evidence"] = [item.as_json() for item in self.evidence]
        return payload


def _join(values: Iterable[Any]) -> str:
    return ";".join(str(value) for value in values if str(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_pairs_path(root: Path, dataset: str, split: str) -> Path:
    candidates = (
        root / "data" / "manifests" / dataset / f"pairs_{split}.csv",
        root / "data" / "manifests" / dataset / "pairs" / f"pairs_{split}.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing canonical pair bundle for {dataset}/{split}: {candidates}")


def _normalize_key_frame(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = dataset
    if "split" not in out.columns:
        out["split"] = split
    if "frgp" not in out.columns and "finger_position" in out.columns:
        out["frgp"] = out["finger_position"]
    for column in ("subject_a", "subject_b", "path_a", "path_b"):
        if column not in out.columns:
            out[column] = ""
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(-1).astype(int)
    out["frgp"] = pd.to_numeric(out["frgp"], errors="coerce").fillna(-1).astype(int)
    out["subject_a"] = out["subject_a"].astype(str).str.strip()
    out["subject_b"] = out["subject_b"].astype(str).str.strip()
    out["path_a"] = out["path_a"].map(_normalize_pair_path)
    out["path_b"] = out["path_b"].map(_normalize_pair_path)
    columns = ["dataset", "split", "pair_id", "label", "frgp", "subject_a", "subject_b", "path_a", "path_b"]
    return out[columns].copy()


def _normalize_pair_path(value: Any) -> str:
    text = str(value).replace("\\", "/").strip()
    lowered = text.lower()
    marker = "/data/"
    index = lowered.find(marker)
    if index >= 0:
        return text[index + 1 :]
    if lowered.startswith("data/"):
        return text
    return text


def load_canonical_bundles(root: Path) -> dict[tuple[str, str], CanonicalBundle]:
    bundles: dict[tuple[str, str], CanonicalBundle] = {}
    for dataset in DATASETS:
        for split in SPLITS:
            path = _canonical_pairs_path(root, dataset, split)
            raw = pd.read_csv(path)
            frame = _normalize_key_frame(raw, dataset=dataset, split=split)
            labels = frame["label"].astype(int)
            frgp_coverage = tuple(sorted(value for value in frame["frgp"].astype(int).unique().tolist() if value > 0))
            bundles[(dataset, split)] = CanonicalBundle(
                dataset=dataset,
                split=split,
                path=path,
                sha256=_file_sha256(path),
                pair_count=int(len(frame)),
                positive_count=int((labels == 1).sum()),
                negative_count=int((labels == 0).sum()),
                frgp_coverage=frgp_coverage,
                frame=frame,
            )
    return bundles


def _infer_dataset_split(path: Path, frame: pd.DataFrame) -> tuple[str | None, str | None]:
    dataset: str | None = None
    split: str | None = None
    if "dataset" in frame.columns:
        values = sorted(str(value).strip() for value in frame["dataset"].dropna().unique().tolist())
        if len(values) == 1 and values[0] in DATASETS:
            dataset = values[0]
    if "split" in frame.columns:
        values = sorted(str(value).strip().lower() for value in frame["split"].dropna().unique().tolist())
        if len(values) == 1 and values[0] in SPLITS:
            split = values[0]
    name = path.name.lower()
    if dataset is None:
        match = re.search(r"(nist_sd300[bc])", name)
        if match:
            dataset = match.group(1)
    if split is None:
        match = re.search(r"(?:^|_)(train|val|test)(?:\.|_|$)", name)
        if match:
            split = match.group(1)
    return dataset, split


def _is_pair_table_header(columns: Iterable[str]) -> bool:
    normalized = {str(column).strip() for column in columns}
    return "pair_id" in normalized and "label" in normalized and ("frgp" in normalized or "finger_position" in normalized)


def _meta_candidates(csv_path: Path) -> tuple[Path, ...]:
    return (
        Path(str(csv_path) + ".meta.json"),
        csv_path.with_suffix(".meta.json"),
    )


def _iter_hash_strings(payload: Any) -> Iterable[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"pair_source_sha256", "selected_pairs_sha256"} and isinstance(value, str):
                yield value
            else:
                yield from _iter_hash_strings(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_hash_strings(item)


def _hashes_from_meta(csv_path: Path) -> set[str]:
    hashes: set[str] = set()
    for candidate in _meta_candidates(csv_path):
        for value in _iter_hash_strings(_read_json(candidate)):
            hashes.add(str(value))
    return hashes


def _hashes_from_score_columns(frame: pd.DataFrame) -> set[str]:
    hashes: set[str] = set()
    for column in ("pair_source_sha256", "selected_pairs_sha256"):
        if column in frame.columns:
            values = {str(value).strip() for value in frame[column].dropna().unique().tolist() if str(value).strip()}
            hashes.update(values)
    return hashes


def _alignment_status(
    frame: pd.DataFrame,
    canonical: CanonicalBundle,
    *,
    table_path: Path,
) -> str:
    actual = _normalize_key_frame(frame, dataset=canonical.dataset, split=canonical.split)
    actual = actual[(actual["dataset"] == canonical.dataset) & (actual["split"] == canonical.split)].copy()
    if int(len(actual)) != canonical.pair_count:
        return f"fail:cannot_hash_row_count:{len(actual)}!=canonical:{canonical.pair_count}"
    if actual["pair_id"].duplicated().any():
        return "fail:duplicate_pair_id"

    if table_path.name.startswith("pairs_") or "/selected_pairs/" in table_path.as_posix().replace("\\", "/"):
        try:
            if _file_sha256(table_path) == canonical.sha256:
                return f"pass:file_sha256:{canonical.sha256}"
        except OSError:
            pass

    meta_hashes = _hashes_from_meta(table_path)
    if canonical.sha256 in meta_hashes:
        return f"pass:metadata_pair_sha256:{canonical.sha256}"

    column_hashes = _hashes_from_score_columns(frame)
    if canonical.sha256 in column_hashes:
        return f"pass:score_column_pair_sha256:{canonical.sha256}"

    key = ["dataset", "split", "pair_id"]
    merged = canonical.frame.merge(
        actual,
        on=key,
        how="outer",
        suffixes=("_canonical", "_artifact"),
        indicator=True,
    )
    missing = int((merged["_merge"] == "left_only").sum())
    extra = int((merged["_merge"] == "right_only").sum())
    if missing or extra:
        return f"fail:pair_id_set_mismatch:missing={missing}:extra={extra}:canonical_sha256={canonical.sha256}"
    both = merged[merged["_merge"] == "both"].copy()
    mismatches = {
        column: int((both[f"{column}_canonical"].astype(str) != both[f"{column}_artifact"].astype(str)).sum())
        for column in ("label", "frgp", "subject_a", "subject_b", "path_a", "path_b")
    }
    bad = {key_: value for key_, value in mismatches.items() if value}
    if bad:
        details = ",".join(f"{key_}={value}" for key_, value in sorted(bad.items()))
        return f"fail:content_mismatch:{details}:canonical_sha256={canonical.sha256}"
    return f"pass:content_alignment_to_canonical_sha256:{canonical.sha256}"


def _validate_group(
    table_path: Path,
    group: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    root: Path,
    canonical_bundles: dict[tuple[str, str], CanonicalBundle],
) -> TableEvidence:
    rel_path = _rel(root, table_path)
    legacy_signals: list[str] = []
    unknown_signals: list[str] = []
    canonical = canonical_bundles.get((dataset, split))
    labels = pd.to_numeric(group["label"], errors="coerce").fillna(-1).astype(int) if "label" in group.columns else None
    frgp_series = None
    if "frgp" in group.columns:
        frgp_series = pd.to_numeric(group["frgp"], errors="coerce").fillna(-1).astype(int)
    elif "finger_position" in group.columns:
        frgp_series = pd.to_numeric(group["finger_position"], errors="coerce").fillna(-1).astype(int)
    frgp_coverage = tuple(sorted(value for value in set(frgp_series.tolist() if frgp_series is not None else []) if value > 0))
    positive_count = int((labels == 1).sum()) if labels is not None else None
    negative_count = int((labels == 0).sum()) if labels is not None else None
    rows = int(len(group))

    if canonical is None:
        row_status = "unknown:no_canonical_bundle"
        label_status = "unknown:no_canonical_bundle"
        frgp_status = "unknown:no_canonical_bundle"
        pair_status = "unknown:no_canonical_bundle"
        canonical_sha = ""
        unknown_signals.append(f"no_canonical_bundle:{dataset}/{split}")
    else:
        canonical_sha = canonical.sha256
        row_status = (
            f"pass:{rows}"
            if rows == canonical.pair_count
            else f"fail:{rows}!=canonical:{canonical.pair_count}"
        )
        label_status = (
            f"pass:pos={positive_count}:neg={negative_count}"
            if positive_count == canonical.positive_count and negative_count == canonical.negative_count
            else (
                "fail:"
                f"pos={positive_count}!=canonical:{canonical.positive_count}:"
                f"neg={negative_count}!=canonical:{canonical.negative_count}"
            )
        )
        frgp_status = (
            "pass:1..10"
            if frgp_coverage == canonical.frgp_coverage == EXPECTED_FRGP
            else f"fail:coverage={','.join(str(value) for value in frgp_coverage)}"
        )
        pair_status = _alignment_status(group, canonical, table_path=table_path)

    if rows in OLD_8_FINGER_COUNTS or positive_count in OLD_8_FINGER_COUNTS or negative_count in OLD_8_FINGER_COUNTS:
        legacy_signals.append("old_8_finger_count_pattern")
    if frgp_coverage and (1 not in frgp_coverage or 6 not in frgp_coverage):
        legacy_signals.append("missing_thumb_frgp_coverage")
    if frgp_coverage and frgp_coverage != EXPECTED_FRGP:
        legacy_signals.append(f"non_v2_frgp_coverage:{','.join(str(value) for value in frgp_coverage)}")
    if pair_status.startswith("fail"):
        legacy_signals.append("pair_hash_or_alignment_mismatch")
    if "/selected_pairs/" in rel_path.replace("\\", "/") and not pair_status.startswith("pass"):
        legacy_signals.append("stale_selected_pairs")
    for token in OLD_TRAIN_SCORE_TOKENS:
        if token.replace("\\", "/").lower() in rel_path.lower():
            legacy_signals.append("old_train_score_path")

    return TableEvidence(
        path=rel_path,
        dataset=dataset,
        split=split,
        rows=rows,
        positive_count=positive_count,
        negative_count=negative_count,
        frgp_coverage=frgp_coverage,
        row_count_status=row_status,
        label_count_status=label_status,
        frgp_status=frgp_status,
        pair_sha_status=pair_status,
        canonical_pair_sha256=canonical_sha,
        legacy_signals=sorted(set(legacy_signals)),
        unknown_signals=sorted(set(unknown_signals)),
    )


def _iter_table_evidence(
    artifact_dir: Path,
    *,
    root: Path,
    canonical_bundles: dict[tuple[str, str], CanonicalBundle],
) -> list[TableEvidence]:
    evidence: list[TableEvidence] = []
    for csv_path in sorted(artifact_dir.rglob("*.csv")):
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception:
            continue
        if not _is_pair_table_header(header.columns):
            continue
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            evidence.append(
                TableEvidence(
                    path=_rel(root, csv_path),
                    dataset="",
                    split="",
                    rows=0,
                    positive_count=None,
                    negative_count=None,
                    frgp_coverage=tuple(),
                    row_count_status="unknown:csv_read_error",
                    label_count_status="unknown:csv_read_error",
                    frgp_status="unknown:csv_read_error",
                    pair_sha_status="unknown:csv_read_error",
                    canonical_pair_sha256="",
                    unknown_signals=[f"csv_read_error:{type(exc).__name__}"],
                )
            )
            continue
        if frame.empty:
            continue
        if "frgp" not in frame.columns and "finger_position" in frame.columns:
            frame = frame.copy()
            frame["frgp"] = frame["finger_position"]

        if "dataset" in frame.columns and "split" in frame.columns:
            frame = frame.copy()
            frame["dataset"] = frame["dataset"].astype(str).str.strip()
            frame["split"] = frame["split"].astype(str).str.strip().str.lower()
            group_keys = [
                (dataset, split)
                for dataset in sorted(frame["dataset"].dropna().unique().tolist())
                for split in sorted(frame.loc[frame["dataset"] == dataset, "split"].dropna().unique().tolist())
                if dataset in DATASETS and split in SPLITS
            ]
            if group_keys:
                for dataset, split in group_keys:
                    group = frame[(frame["dataset"] == dataset) & (frame["split"] == split)].copy()
                    evidence.append(
                        _validate_group(
                            csv_path,
                            group,
                            dataset=dataset,
                            split=split,
                            root=root,
                            canonical_bundles=canonical_bundles,
                        )
                    )
                continue

        dataset, split = _infer_dataset_split(csv_path, frame)
        if dataset is None or split is None:
            evidence.append(
                TableEvidence(
                    path=_rel(root, csv_path),
                    dataset=dataset or "",
                    split=split or "",
                    rows=int(len(frame)),
                    positive_count=None,
                    negative_count=None,
                    frgp_coverage=tuple(),
                    row_count_status="unknown:cannot_infer_dataset_split",
                    label_count_status="unknown:cannot_infer_dataset_split",
                    frgp_status="unknown:cannot_infer_dataset_split",
                    pair_sha_status="unknown:cannot_infer_dataset_split",
                    canonical_pair_sha256="",
                    unknown_signals=["cannot_infer_dataset_split"],
                )
            )
            continue
        evidence.append(
            _validate_group(
                csv_path,
                frame,
                dataset=dataset,
                split=split,
                root=root,
                canonical_bundles=canonical_bundles,
            )
        )
    return evidence


def _status_summary(evidence: list[TableEvidence], attr: str) -> str:
    if not evidence:
        return "n/a:no_pair_tables"
    counts = Counter(getattr(item, attr).split(":", 1)[0] for item in evidence)
    parts = [f"{status}={counts[status]}" for status in sorted(counts)]
    return ",".join(parts)


def _text_contains_legacy_signal(path: Path) -> list[str]:
    signals: list[str] = []
    try:
        if path.stat().st_size > MAX_TEXT_SCAN_BYTES:
            return signals
        text = path.read_text(encoding="utf-8", errors="ignore").replace("\\", "/")
    except OSError:
        return signals
    lowered = text.lower()
    for token in OLD_TRAIN_SCORE_TOKENS:
        normalized = token.replace("\\", "/").lower()
        start = lowered.find(normalized)
        if start >= 0 and not _is_negative_reference_context(lowered, start):
            signals.append("old_train_score_path")
    return signals


def _artifact_text_signals(artifact_dir: Path) -> list[str]:
    signals: list[str] = []
    for path in artifact_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            signals.extend(_text_contains_legacy_signal(path))
    return sorted(set(signals))


def _all_passed(evidence: list[TableEvidence]) -> bool:
    return bool(evidence) and all(item.passed_v2 for item in evidence)


def _evidence_pairs(evidence: list[TableEvidence]) -> set[tuple[str, str]]:
    return {(item.dataset, item.split) for item in evidence if item.dataset and item.split}


def _missing_required_pairs(evidence: list[TableEvidence], required: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return set(required) - _evidence_pairs(evidence)


def _recommendation(classification: str, *, final_failed: bool) -> str:
    if classification == CLASS_FINAL:
        return "allowed_for_final_or_pre_final_inputs"
    if classification == CLASS_COMPAT:
        return "allowed_as_current_v2_compatibility_input"
    if classification == CLASS_LEGACY:
        return "warning_only_keep_for_provenance_do_not_use_as_input"
    if final_failed:
        return "block_final_use_until_content_validation_passes"
    return "manual_review_required_before_any_use"


def classify_artifact(
    artifact_dir: Path,
    *,
    root: Path,
    canonical_bundles: dict[tuple[str, str], CanonicalBundle],
) -> ArtifactAuditRow:
    name = artifact_dir.name
    rel_path = _rel(root, artifact_dir)
    evidence = _iter_table_evidence(artifact_dir, root=root, canonical_bundles=canonical_bundles)
    datasets = tuple(sorted({item.dataset for item in evidence if item.dataset}))
    splits = tuple(sorted({item.split for item in evidence if item.split}, key=lambda value: SPLITS.index(value) if value in SPLITS else 99))
    legacy_signals = sorted(
        {
            signal
            for item in evidence
            for signal in item.legacy_signals
        }
        | set(_artifact_text_signals(artifact_dir))
    )
    unknown_signals = sorted({signal for item in evidence for signal in item.unknown_signals})
    row_status = _status_summary(evidence, "row_count_status")
    label_status = _status_summary(evidence, "label_count_status")
    frgp_status = _status_summary(evidence, "frgp_status")
    pair_status = _status_summary(evidence, "pair_sha_status")

    is_final_allowlisted = name in FINAL_ALLOWLIST_DIRS
    final_failed = False
    reason = ""

    if name in KNOWN_LEGACY_DIRS:
        classification = CLASS_LEGACY
        reason = "known legacy benchmark artifact directory; warning-only inventory entry"
        legacy_signals = sorted(set(legacy_signals + ["known_legacy_directory_name"]))
    elif name in COMPATIBILITY_CANDIDATE_DIRS:
        required = set(COMPATIBILITY_CANDIDATE_DIRS[name]["required"])
        missing = _missing_required_pairs(evidence, required)
        if _all_passed(evidence) and not missing:
            classification = CLASS_COMPAT
            reason = "compatibility directory validated against canonical v2 pair bundles"
        else:
            if missing:
                unknown_signals.append(
                    "missing_required_score_tables:" + ",".join(f"{dataset}/{split}" for dataset, split in sorted(missing))
                )
            if legacy_signals:
                classification = CLASS_LEGACY
                reason = "compatibility candidate failed v2 validation and shows legacy content signals"
            else:
                classification = CLASS_UNKNOWN
                reason = "compatibility candidate did not satisfy required v2 score validation"
    elif is_final_allowlisted:
        if name in FINAL_NON_PAIR_ALLOWED_DIRS and not evidence:
            classification = CLASS_FINAL
            reason = "final allowlisted non-pair report/package directory"
        elif _all_passed(evidence):
            classification = CLASS_FINAL
            reason = "final allowlisted artifact with canonical v2 content validation"
        else:
            final_failed = True
            classification = CLASS_UNKNOWN
            if not evidence:
                unknown_signals.append("final_allowlisted_dir_has_no_pair_tables")
            reason = "final allowlisted artifact failed canonical v2 content validation"
    elif evidence and _all_passed(evidence):
        classification = CLASS_COMPAT
        reason = "unlisted benchmark directory contains v2-aligned score or pair tables"
    elif legacy_signals:
        classification = CLASS_LEGACY
        reason = "content-derived legacy signals detected"
    else:
        classification = CLASS_UNKNOWN
        if not evidence:
            unknown_signals.append("no_pair_or_score_tables_found")
        reason = "unrecognized benchmark artifact directory"

    legacy_signals = tuple(sorted(set(legacy_signals)))
    unknown_signals = tuple(sorted(set(unknown_signals)))
    return ArtifactAuditRow(
        artifact_path=rel_path,
        classification=classification,
        do_not_use=classification in {CLASS_LEGACY, CLASS_UNKNOWN},
        is_final_allowed=classification == CLASS_FINAL and is_final_allowlisted,
        reason=reason,
        datasets_found=datasets,
        splits_found=splits,
        row_count_status=row_status,
        label_count_status=label_status,
        frgp_status=frgp_status,
        pair_sha_status=pair_status,
        legacy_signals=legacy_signals,
        unknown_signals=unknown_signals,
        recommended_action=_recommendation(classification, final_failed=final_failed),
        evidence=evidence,
    )


def _discover_artifact_dirs(root: Path, outdir: Path) -> list[Path]:
    benchmark_root = root / BENCHMARK_ROOT_REL
    if not benchmark_root.exists():
        return []
    out_resolved = outdir.resolve()
    dirs = []
    for child in sorted(benchmark_root.iterdir()):
        if child.is_dir() and child.resolve() != out_resolved:
            dirs.append(child)
    return dirs


def _is_boundary(value: str) -> bool:
    return value == "" or not (value.isalnum() or value in {"_", "-", "."})


def _token_occurrences(text: str, token: str) -> Iterable[int]:
    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            return
        before = text[index - 1 : index] if index > 0 else ""
        after = text[index + len(token) : index + len(token) + 1]
        if _is_boundary(before) and _is_boundary(after):
            yield index
        start = index + 1


def _is_negative_reference_context(lowered_text: str, index: int) -> bool:
    context = lowered_text[max(0, index - 300) : index + 120]
    markers = (
        "forbidden_inputs",
        "forbidden input",
        "forbidden path",
        "disallowed",
        "do not use",
        "do_not_use",
        "not used",
        "warning-only",
        "warning only",
        "legacy candidate",
        "legacy_signals",
        "known_legacy",
        "audit",
    )
    return any(marker in context for marker in markers)


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def scan_script_references(root: Path, tokens: Iterable[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    normalized_tokens = tuple(sorted({token.replace("\\", "/").rstrip("/") for token in tokens if token}, key=len, reverse=True))
    for rel_path in FINAL_PREFINAL_SCAN_FILES:
        path = root / rel_path
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").replace("\\", "/")
        except OSError:
            continue
        lowered = text.lower()
        lines = text.splitlines()
        seen: set[tuple[str, int]] = set()
        for token in normalized_tokens:
            token_lower = token.lower()
            for index in _token_occurrences(lowered, token_lower):
                if _is_negative_reference_context(lowered, index):
                    continue
                line_no = _line_number(text, index)
                key = (token, line_no)
                if key in seen:
                    continue
                seen.add(key)
                line = lines[line_no - 1].strip() if 0 <= line_no - 1 < len(lines) else ""
                hits.append(
                    {
                        "file": rel_path,
                        "line": line_no,
                        "artifact": token,
                        "line_text": line[:300],
                    }
                )
    return hits


def write_inventory_csv(path: Path, rows: list[ArtifactAuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    counts = payload["classification_counts"]
    lines = [
        "# Final SD300 Anatomical v2 Forensic Artifact Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- created_at: `{payload['created_at']}`",
        f"- repo_root: `{payload['repo_root']}`",
        f"- destructive_actions_performed: `{str(payload['destructive_actions_performed']).lower()}`",
        f"- training_ran: `{str(payload['training_ran']).lower()}`",
        f"- fusion_reran: `{str(payload['fusion_reran']).lower()}`",
        f"- kaggle_ran: `{str(payload['kaggle_ran']).lower()}`",
        f"- pair_bundles_modified: `{str(payload['pair_bundles_modified']).lower()}`",
        "",
        "## Classification Counts",
        "",
    ]
    for key in (CLASS_FINAL, CLASS_COMPAT, CLASS_LEGACY, CLASS_UNKNOWN):
        lines.append(f"- {key}: `{counts.get(key, 0)}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- inventory: `{payload['output_paths']['inventory_csv']}`",
            f"- summary: `{payload['output_paths']['summary_md']}`",
            f"- manifest: `{payload['output_paths']['manifest_json']}`",
            "",
            "## Failure Gates",
            "",
            f"- final allowlisted content failures: `{len(payload['final_allowlist_failures'])}`",
            f"- legacy references in final/pre-final scripts: `{len(payload['legacy_reference_hits'])}`",
            f"- unknown artifacts referenced by final/pre-final scripts: `{len(payload['unknown_reference_hits'])}`",
        ]
    )
    if payload["final_allowlist_failures"]:
        lines.extend(["", "### Final Allowlist Failures", ""])
        for item in payload["final_allowlist_failures"]:
            lines.append(f"- `{item['artifact_path']}`: {item['reason']}")
    if payload["legacy_reference_hits"]:
        lines.extend(["", "### Legacy Reference Hits", ""])
        for hit in payload["legacy_reference_hits"]:
            lines.append(f"- `{hit['file']}:{hit['line']}` references `{hit['artifact']}`")
    if payload["unknown_reference_hits"]:
        lines.extend(["", "### Unknown Reference Hits", ""])
        for hit in payload["unknown_reference_hits"]:
            lines.append(f"- `{hit['file']}:{hit['line']}` references `{hit['artifact']}`")
    lines.extend(["", "## Inventory", ""])
    for row in payload["inventory"]:
        lines.append(
            "- `{classification}` `{artifact_path}` do_not_use=`{do_not_use}` final_allowed=`{is_final_allowed}`".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(root: Path, outdir: Path) -> tuple[dict[str, Any], int]:
    canonical_bundles = load_canonical_bundles(root)
    rows = [
        classify_artifact(path, root=root, canonical_bundles=canonical_bundles)
        for path in _discover_artifact_dirs(root, outdir)
    ]
    counts = Counter(row.classification for row in rows)
    final_allowlist_failures = [
        row.as_json()
        for row in rows
        if row.artifact_path.rsplit("/", 1)[-1] in FINAL_ALLOWLIST_DIRS and row.classification != CLASS_FINAL
    ]
    legacy_reference_hits = scan_script_references(root, LEGACY_REFERENCE_TOKENS)
    unknown_tokens = [
        row.artifact_path.rsplit("/", 1)[-1]
        for row in rows
        if row.classification == CLASS_UNKNOWN
    ]
    unknown_reference_hits = scan_script_references(root, unknown_tokens)

    exit_code = 0
    if final_allowlist_failures or legacy_reference_hits or unknown_reference_hits:
        exit_code = 2

    status = "pass" if exit_code == 0 else "fail"
    inventory_csv = outdir / "forensic_artifact_inventory.csv"
    summary_md = outdir / "forensic_artifact_audit_summary.md"
    manifest_json = outdir / "forensic_artifact_audit_manifest.json"
    payload = {
        "schema_version": "final_sd300_anatomical_v2_forensic_artifact_audit_v2",
        "created_at": _utc_now(),
        "repo_root": str(root),
        "outdir": str(outdir),
        "status": status,
        "exit_code": exit_code,
        "destructive_actions_performed": False,
        "training_ran": False,
        "fusion_reran": False,
        "kaggle_ran": False,
        "pair_bundles_modified": False,
        "scan_scope": {
            "benchmark_root": str(root / BENCHMARK_ROOT_REL),
            "direct_child_directories_only": True,
            "excluded_self_audit_output": str(outdir),
            "final_prefinal_scan_files": list(FINAL_PREFINAL_SCAN_FILES),
        },
        "canonical_pair_bundles": [
            {
                "dataset": item.dataset,
                "split": item.split,
                "path": str(item.path),
                "sha256": item.sha256,
                "pair_count": item.pair_count,
                "positive_count": item.positive_count,
                "negative_count": item.negative_count,
                "frgp_coverage": list(item.frgp_coverage),
            }
            for item in canonical_bundles.values()
        ],
        "classification_counts": {key: int(counts.get(key, 0)) for key in (CLASS_FINAL, CLASS_COMPAT, CLASS_LEGACY, CLASS_UNKNOWN)},
        "final_allowlist_failures": final_allowlist_failures,
        "legacy_reference_hits": legacy_reference_hits,
        "unknown_reference_hits": unknown_reference_hits,
        "inventory": [row.as_json() for row in rows],
        "output_paths": {
            "inventory_csv": str(inventory_csv),
            "summary_md": str(summary_md),
            "manifest_json": str(manifest_json),
        },
    }

    outdir.mkdir(parents=True, exist_ok=True)
    write_inventory_csv(inventory_csv, rows)
    write_summary(summary_md, payload)
    manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload, exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forensically classify SD300 benchmark artifact directories without modifying pair bundles."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Audit report output directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.repo_root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir
    payload, exit_code = run_audit(root, outdir.resolve())
    counts = payload["classification_counts"]
    print(f"Forensic artifact audit {payload['status']}.")
    print(f"inventory: {payload['output_paths']['inventory_csv']}")
    print(f"summary: {payload['output_paths']['summary_md']}")
    print(f"manifest: {payload['output_paths']['manifest_json']}")
    print(
        "classification_counts: "
        + ", ".join(f"{key}={counts.get(key, 0)}" for key in (CLASS_FINAL, CLASS_COMPAT, CLASS_LEGACY, CLASS_UNKNOWN))
    )
    if payload["final_allowlist_failures"]:
        print("Final allowlisted artifact validation failures:", file=sys.stderr)
        for item in payload["final_allowlist_failures"]:
            print(f"- {item['artifact_path']}: {item['reason']}", file=sys.stderr)
    if payload["legacy_reference_hits"]:
        print("Legacy candidate references in final/pre-final scripts:", file=sys.stderr)
        for hit in payload["legacy_reference_hits"]:
            print(f"- {hit['file']}:{hit['line']} -> {hit['artifact']}", file=sys.stderr)
    if payload["unknown_reference_hits"]:
        print("Unknown artifacts referenced by final/pre-final scripts:", file=sys.stderr)
        for hit in payload["unknown_reference_hits"]:
            print(f"- {hit['file']}:{hit['line']} -> {hit['artifact']}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
