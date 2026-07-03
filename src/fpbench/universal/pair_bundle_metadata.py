from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


SD300_DATASETS = {"nist_sd300b", "nist_sd300c"}
SD300_RUN_PAIR_BUNDLE_VERSION = "sd300_anatomical_full_pairs_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def is_artifact_selected_pairs_path(path: Path, *, repo_root: Path) -> bool:
    resolved = path.resolve()
    reports_root = (repo_root / "artifacts" / "reports").resolve()
    return is_relative_to(resolved, reports_root) and "selected_pairs" in {part.lower() for part in resolved.parts}


def manifest_path_for_pair_source(*, dataset: str, pairs_csv: Path, repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "data" / "manifests" / dataset / "manifest.csv",
        repo_root / "data" / "processed" / dataset / "manifest.csv",
    ]
    parent = pairs_csv.resolve().parent
    candidates.extend([parent / "manifest.csv", parent.parent / "manifest.csv"])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def split_subjects_path_for_pair_source(*, dataset: str, pairs_csv: Path, repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "data" / "manifests" / dataset / "pairs" / "split_subjects.json",
        repo_root / "data" / "processed" / dataset / "pairs" / "split_subjects.json",
    ]
    parent = pairs_csv.resolve().parent
    candidates.extend([parent / "split_subjects.json", parent.parent / "pairs" / "split_subjects.json"])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _label_count(labels: pd.Series, label: int) -> int:
    numeric = pd.to_numeric(labels, errors="coerce").fillna(-1).astype(int)
    return int((numeric == int(label)).sum())


def _frgp_counts(pairs: pd.DataFrame) -> dict[str, int]:
    if "frgp" not in pairs.columns:
        return {}
    values = pd.to_numeric(pairs["frgp"], errors="coerce").dropna().astype(int)
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def build_pair_bundle_metadata(
    *,
    dataset: str,
    split: str,
    pair_source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    pairs_path = pair_source_path.resolve()
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pair source does not exist: {pairs_path}")

    pairs = pd.read_csv(pairs_path)
    manifest_path = manifest_path_for_pair_source(dataset=dataset, pairs_csv=pairs_path, repo_root=repo_root)
    split_subjects_path = split_subjects_path_for_pair_source(dataset=dataset, pairs_csv=pairs_path, repo_root=repo_root)
    manifest_columns: set[str] = set()
    if manifest_path is not None and manifest_path.exists():
        try:
            manifest_columns = set(pd.read_csv(manifest_path, nrows=0).columns)
        except Exception:
            manifest_columns = set()

    labels = pairs["label"] if "label" in pairs.columns else pd.Series(dtype=int)
    is_sd300 = dataset in SD300_DATASETS
    return {
        "dataset_id": dataset,
        "split": split,
        "pair_source_path": str(pairs_path),
        "pair_source_sha256": file_sha256(pairs_path),
        "manifest_source_path": str(manifest_path) if manifest_path is not None else "",
        "manifest_source_sha256": file_sha256(manifest_path) if manifest_path is not None and manifest_path.exists() else "",
        "split_subjects_path": str(split_subjects_path) if split_subjects_path is not None else "",
        "split_subjects_sha256": file_sha256(split_subjects_path)
        if split_subjects_path is not None and split_subjects_path.exists()
        else "",
        "pair_count": int(len(pairs)),
        "positive_count": _label_count(labels, 1),
        "negative_count": _label_count(labels, 0),
        "frgp_counts": _frgp_counts(pairs),
        "sd300_frgp_semantics": "anatomical" if is_sd300 else "dataset_native",
        "sd300_raw_frgp_available": bool(is_sd300 and "raw_frgp" in manifest_columns),
        "run_pair_bundle_version": SD300_RUN_PAIR_BUNDLE_VERSION if is_sd300 else "",
    }
