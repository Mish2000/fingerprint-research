from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd
import yaml

from pipelines.ingest.pair_bundle_utils import (
    CANONICAL_PAIR_COLUMNS,
    CANONICAL_PAIR_SCHEMA_VERSION,
    PAIR_BUILD_META_SCHEMA_VERSION,
    REQUIRED_PAIR_BUILD_META_FIELDS,
    REQUIRED_SPLIT_SUBJECTS_FIELDS,
    SPLIT_SUBJECTS_SCHEMA_VERSION,
    validate_canonical_pairs_df,
    validate_pairs_split_build_meta,
    validate_split_subjects_metadata,
)


DEFAULT_SPLITS = ("train", "val", "test")


@dataclass
class ValidationReport:
    root: Path
    errors: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"  [FAIL] {message}")

    def ok(self, message: str) -> None:
        print(f"  [OK] {message}")

    @property
    def failed(self) -> bool:
        return bool(self.errors)



def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)



def load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML at {path} must decode to a mapping")
    return payload



def norm_capture(value: Any) -> str:
    s = str(value).strip().lower()
    aliases = {
        "p": "plain",
        "r": "roll",
        "rolled": "roll",
        "contactless": "contactless",
        "contact-less": "contactless",
        "contact_less": "contactless",
        "contactbased": "contact_based",
        "contact-based": "contact_based",
        "contact_based": "contact_based",
    }
    return aliases.get(s, s)



def resolve_root(root_arg: str | None) -> Path:
    if root_arg:
        resolved = Path(root_arg).expanduser().resolve()
        if (resolved / "configs" / "datasets.yaml").exists():
            return resolved
        raise FileNotFoundError(
            f"Could not resolve project root from --root={resolved}. "
            "Expected configs/datasets.yaml under that directory."
        )

    env_root = os.environ.get("FPRJ_ROOT")
    if env_root:
        resolved = Path(env_root).expanduser().resolve()
        if (resolved / "configs" / "datasets.yaml").exists():
            return resolved
        raise FileNotFoundError(
            f"Could not resolve project root from FPRJ_ROOT={resolved}. "
            "Expected configs/datasets.yaml under that directory."
        )

    resolved = Path(__file__).resolve().parent
    if (resolved / "configs" / "datasets.yaml").exists():
        return resolved
    raise FileNotFoundError(
        f"Could not resolve project root from script location {resolved}. "
        "Provide --root <repo>, or set FPRJ_ROOT to a valid repository root."
    )



def _expect_mapping(payload: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return payload



def _active_datasets(config: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    datasets = _expect_mapping(config.get("datasets"), context="configs/datasets.yaml datasets")
    active = {
        str(name): _expect_mapping(spec, context=f"dataset {name!r}")
        for name, spec in datasets.items()
        if isinstance(spec, Mapping) and str(spec.get("status", "")).lower() == "active"
    }
    if not active:
        raise ValueError("configs/datasets.yaml contains no active datasets")
    return active



def _validate_dataset_registry(config: Mapping[str, Any]) -> None:
    schemas = _expect_mapping(config.get("schemas"), context="configs/datasets.yaml schemas")

    pairs_csv = _expect_mapping(schemas.get("pairs_csv"), context="schemas.pairs_csv")
    required_pair_columns = list(pairs_csv.get("required_columns", []))
    if required_pair_columns != CANONICAL_PAIR_COLUMNS:
        raise ValueError(
            "configs/datasets.yaml schemas.pairs_csv.required_columns must match canonical pair columns; "
            f"found {required_pair_columns!r}"
        )
    if str(pairs_csv.get("schema_version")) != CANONICAL_PAIR_SCHEMA_VERSION:
        raise ValueError(
            "configs/datasets.yaml schemas.pairs_csv.schema_version must be "
            f"{CANONICAL_PAIR_SCHEMA_VERSION!r}; found {pairs_csv.get('schema_version')!r}"
        )

    split_schema = _expect_mapping(schemas.get("pairs_split_subjects_json"), context="schemas.pairs_split_subjects_json")
    split_required = list(split_schema.get("required_fields", []))
    if split_required != REQUIRED_SPLIT_SUBJECTS_FIELDS:
        raise ValueError(
            "configs/datasets.yaml schemas.pairs_split_subjects_json.required_fields must match validator requirements; "
            f"found {split_required!r}"
        )
    if str(split_schema.get("schema_version")) != SPLIT_SUBJECTS_SCHEMA_VERSION:
        raise ValueError(
            "configs/datasets.yaml schemas.pairs_split_subjects_json.schema_version must be "
            f"{SPLIT_SUBJECTS_SCHEMA_VERSION!r}; found {split_schema.get('schema_version')!r}"
        )
    if str(split_schema.get("canonical_pair_schema_version")) != CANONICAL_PAIR_SCHEMA_VERSION:
        raise ValueError(
            "configs/datasets.yaml schemas.pairs_split_subjects_json.canonical_pair_schema_version must be "
            f"{CANONICAL_PAIR_SCHEMA_VERSION!r}; found {split_schema.get('canonical_pair_schema_version')!r}"
        )

    build_schema = _expect_mapping(schemas.get("pairs_split_build_meta_json"), context="schemas.pairs_split_build_meta_json")
    build_required = list(build_schema.get("required_fields", []))
    if build_required != REQUIRED_PAIR_BUILD_META_FIELDS:
        raise ValueError(
            "configs/datasets.yaml schemas.pairs_split_build_meta_json.required_fields must match validator requirements; "
            f"found {build_required!r}"
        )
    if str(build_schema.get("schema_version")) != PAIR_BUILD_META_SCHEMA_VERSION:
        raise ValueError(
            "configs/datasets.yaml schemas.pairs_split_build_meta_json.schema_version must be "
            f"{PAIR_BUILD_META_SCHEMA_VERSION!r}; found {build_schema.get('schema_version')!r}"
        )
    if str(build_schema.get("canonical_pair_schema_version")) != CANONICAL_PAIR_SCHEMA_VERSION:
        raise ValueError(
            "configs/datasets.yaml schemas.pairs_split_build_meta_json.canonical_pair_schema_version must be "
            f"{CANONICAL_PAIR_SCHEMA_VERSION!r}; found {build_schema.get('canonical_pair_schema_version')!r}"
        )



def _resolve_dataset_manifest_dir(root: Path, dataset_name: str, dataset_spec: Mapping[str, Any]) -> Path:
    manifests = _expect_mapping(dataset_spec.get("manifests"), context=f"{dataset_name} manifests")
    proposed = manifests.get("proposed")
    if proposed:
        base = root / str(proposed)
    else:
        base = root / "data" / "manifests" / dataset_name
    return base



def _pairs_candidates(base: Path, split_name: str) -> Iterable[Path]:
    yield base / "pairs" / f"pairs_{split_name}.csv"
    yield base / f"pairs_{split_name}.csv"



def _find_existing_path(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None



def _require_file(path: Path, *, label: str, report: ValidationReport) -> bool:
    if path.exists():
        report.ok(f"{label}: {path}")
        return True
    report.error(f"missing required file: {path} ({label})")
    return False



def _validate_manifest(manifest_path: Path, *, dataset_name: str, report: ValidationReport) -> pd.DataFrame | None:
    if not _require_file(manifest_path, label="manifest.csv", report=report):
        return None

    df = pd.read_csv(manifest_path)
    required_cols = ["dataset", "capture", "subject_id", "impression", "ppi", "frgp", "path", "split"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        report.error(f"manifest.csv missing required columns: {missing}")
        return None

    if df[required_cols].isnull().any().any():
        null_counts = {k: int(v) for k, v in df[required_cols].isnull().sum().items() if int(v) > 0}
        report.error(f"manifest.csv contains nulls in required fields: {null_counts}")
        return None

    manifest_dataset_values = sorted(df["dataset"].astype(str).unique().tolist())
    if manifest_dataset_values != [dataset_name]:
        report.error(
            f"manifest.csv dataset column must contain only {dataset_name!r}; found {manifest_dataset_values}"
        )
        return None

    split_values = sorted(df["split"].astype(str).unique().tolist())
    if split_values != sorted(DEFAULT_SPLITS):
        report.error(f"manifest.csv split column must contain train/val/test; found {split_values}")
        return None

    df = df.copy()
    df["capture_norm"] = df["capture"].map(norm_capture)
    report.ok(f"manifest.csv rows={len(df)} capture_values={sorted(df['capture_norm'].unique().tolist())}")
    return df



def _validate_pairs_csv(
    pairs_path: Path,
    *,
    split_name: str,
    manifest_df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    if not _require_file(pairs_path, label=f"pairs_{split_name}.csv", report=report):
        return

    pairs = pd.read_csv(pairs_path)
    try:
        pairs = validate_canonical_pairs_df(
            pairs,
            context=str(pairs_path),
            expected_split=split_name,
            require_exact_columns=True,
            require_non_empty=True,
        )
    except ValueError as exc:
        report.error(str(exc))
        return

    manifest_meta = manifest_df[["path", "subject_id", "frgp", "split"]].drop_duplicates(subset=["path"]).set_index("path")
    joined = pairs.join(manifest_meta.add_prefix("a_"), on="path_a").join(manifest_meta.add_prefix("b_"), on="path_b")

    missing_a = int(joined["a_subject_id"].isna().sum())
    missing_b = int(joined["b_subject_id"].isna().sum())
    if missing_a or missing_b:
        report.error(
            f"{pairs_path} references paths missing from manifest: path_a_missing={missing_a}, path_b_missing={missing_b}"
        )
        return

    split_mismatch_a = int((joined["a_split"].astype(str) != split_name).sum())
    split_mismatch_b = int((joined["b_split"].astype(str) != split_name).sum())
    if split_mismatch_a or split_mismatch_b:
        report.error(
            f"{pairs_path} manifest split mismatch: path_a={split_mismatch_a}, path_b={split_mismatch_b}"
        )
        return

    subject_mismatch_a = int((joined["subject_a"].astype(int) != joined["a_subject_id"].astype(int)).sum())
    subject_mismatch_b = int((joined["subject_b"].astype(int) != joined["b_subject_id"].astype(int)).sum())
    frgp_mismatch = int(
        ((joined["frgp"].astype(int) != joined["a_frgp"].astype(int)) | (joined["frgp"].astype(int) != joined["b_frgp"].astype(int))).sum()
    )
    if subject_mismatch_a or subject_mismatch_b or frgp_mismatch:
        report.error(
            f"{pairs_path} canonical fields disagree with manifest metadata: "
            f"subject_a={subject_mismatch_a}, subject_b={subject_mismatch_b}, frgp={frgp_mismatch}"
        )
        return

    label_counts = pairs["label"].value_counts(dropna=False).to_dict()
    report.ok(f"{pairs_path.name} rows={len(pairs)} label_counts={label_counts}")



def _validate_split_subjects(path: Path, *, report: ValidationReport) -> None:
    if not _require_file(path, label="pairs/split_subjects.json", report=report):
        return
    try:
        payload = load_json(path)
        validate_split_subjects_metadata(payload, context=str(path))
    except (ValueError, json.JSONDecodeError) as exc:
        report.error(str(exc))
        return
    report.ok(f"split_subjects metadata valid (schema_version={payload['schema_version']})")



def _validate_pairs_build_meta(path: Path, *, dataset_name: str, report: ValidationReport) -> None:
    if not _require_file(path, label="pairs_split_build.meta.json", report=report):
        return
    try:
        payload = load_json(path)
        validate_pairs_split_build_meta(payload, context=str(path))
    except (ValueError, json.JSONDecodeError) as exc:
        report.error(str(exc))
        return

    if str(payload.get("dataset")) != dataset_name:
        report.error(f"{path} dataset must be {dataset_name!r}; found {payload.get('dataset')!r}")
        return

    report.ok(f"pairs_split_build meta valid (schema_version={payload['schema_version']})")



def validate_repository(root: Path) -> int:
    print(f"Repository root: {root}")
    config_path = root / "configs" / "datasets.yaml"
    if not config_path.exists():
        print(f"[FAIL] Missing required config: {config_path}")
        return 2

    try:
        config = load_yaml(config_path)
        _validate_dataset_registry(config)
        active_datasets = _active_datasets(config)
    except Exception as exc:
        print(f"[FAIL] Invalid dataset registry: {exc}")
        return 2

    overall_errors = 0
    for dataset_name, dataset_spec in active_datasets.items():
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        base = _resolve_dataset_manifest_dir(root, dataset_name, dataset_spec)
        report = ValidationReport(root=root)
        report.ok(f"manifest root: {base}")

        manifest_df = _validate_manifest(base / "manifest.csv", dataset_name=dataset_name, report=report)
        if manifest_df is not None:
            for split_name in DEFAULT_SPLITS:
                pairs_path = _find_existing_path(_pairs_candidates(base, split_name))
                if pairs_path is None:
                    report.error(
                        f"missing required pairs CSV for split {split_name!r}; checked "
                        f"{[str(p) for p in _pairs_candidates(base, split_name)]}"
                    )
                    continue
                _validate_pairs_csv(
                    pairs_path,
                    split_name=split_name,
                    manifest_df=manifest_df,
                    report=report,
                )

        _validate_split_subjects(base / "pairs" / "split_subjects.json", report=report)
        _validate_pairs_build_meta(base / "pairs_split_build.meta.json", dataset_name=dataset_name, report=report)

        if report.failed:
            overall_errors += len(report.errors)
            print(f"SUMMARY: FAILED with {len(report.errors)} issue(s)")
        else:
            print("SUMMARY: PASS")

    if overall_errors:
        print("\n" + "=" * 100)
        print(f"VALIDATION FAILED: {overall_errors} issue(s) detected across active datasets")
        return 1

    print("\n" + "=" * 100)
    print("VALIDATION PASSED: all active datasets satisfy the canonical pair bundle requirements")
    return 0



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate active dataset manifests and pair bundles against the canonical pair schema. "
            "Supports --root <repo> or FPRJ_ROOT for portable execution."
        )
    )
    parser.add_argument("--root", help="Path to the repository root. Overrides auto-detection.")
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = resolve_root(args.root)
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    return validate_repository(root)


if __name__ == "__main__":
    raise SystemExit(main())
