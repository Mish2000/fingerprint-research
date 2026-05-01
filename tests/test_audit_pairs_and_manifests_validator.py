from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from pipelines.ingest.pair_bundle_utils import (
    CANONICAL_PAIR_COLUMNS,
    CANONICAL_PAIR_SCHEMA_VERSION,
    PAIR_BUILD_META_SCHEMA_VERSION,
    REQUIRED_PAIR_BUILD_META_FIELDS,
    REQUIRED_SPLIT_SUBJECTS_FIELDS,
    SPLIT_SUBJECTS_SCHEMA_VERSION,
    build_pairs_split_build_meta,
    build_split_subjects_metadata,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "validate_pairs_and_manifests.py"
DATASET_NAME = "toyset"



def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")



def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def _build_dataset_registry() -> dict:
    return {
        "version": 1,
        "schema_version": "v2_dataset_registry",
        "schemas": {
            "pairs_csv": {
                "required_columns": list(CANONICAL_PAIR_COLUMNS),
                "schema_version": CANONICAL_PAIR_SCHEMA_VERSION,
            },
            "pairs_split_subjects_json": {
                "schema_version": SPLIT_SUBJECTS_SCHEMA_VERSION,
                "required_fields": list(REQUIRED_SPLIT_SUBJECTS_FIELDS),
                "canonical_pair_schema_version": CANONICAL_PAIR_SCHEMA_VERSION,
            },
            "pairs_split_build_meta_json": {
                "schema_version": PAIR_BUILD_META_SCHEMA_VERSION,
                "required_fields": list(REQUIRED_PAIR_BUILD_META_FIELDS),
                "canonical_pair_schema_version": CANONICAL_PAIR_SCHEMA_VERSION,
            },
        },
        "datasets": {
            DATASET_NAME: {
                "status": "active",
                "manifests": {"proposed": f"data/manifests/{DATASET_NAME}"},
            }
        },
    }



def _manifest_rows() -> list[dict]:
    rows: list[dict] = []
    split_subjects = {
        "train": (1, 2),
        "val": (3, 4),
        "test": (5, 6),
    }
    for split, subjects in split_subjects.items():
        for subject_id in subjects:
            for impression in (1, 2):
                rows.append(
                    {
                        "dataset": DATASET_NAME,
                        "capture": "plain",
                        "subject_id": subject_id,
                        "impression": f"sample_{impression}",
                        "ppi": 500,
                        "frgp": 1,
                        "path": f"data/raw/{DATASET_NAME}/{split}/s{subject_id}_i{impression}.png",
                        "split": split,
                    }
                )
    return rows



def _pairs_rows(split: str, subject_a: int, subject_b: int) -> list[dict]:
    return [
        {
            "pair_id": 0,
            "label": 1,
            "split": split,
            "subject_a": subject_a,
            "subject_b": subject_a,
            "frgp": 1,
            "path_a": f"data/raw/{DATASET_NAME}/{split}/s{subject_a}_i1.png",
            "path_b": f"data/raw/{DATASET_NAME}/{split}/s{subject_a}_i2.png",
        },
        {
            "pair_id": 1,
            "label": 0,
            "split": split,
            "subject_a": subject_a,
            "subject_b": subject_b,
            "frgp": 1,
            "path_a": f"data/raw/{DATASET_NAME}/{split}/s{subject_a}_i1.png",
            "path_b": f"data/raw/{DATASET_NAME}/{split}/s{subject_b}_i1.png",
        },
    ]



def _write_valid_repo(root: Path) -> Path:
    _write_yaml(root / "configs" / "datasets.yaml", _build_dataset_registry())

    base = root / "data" / "manifests" / DATASET_NAME
    pairs_dir = base / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(_manifest_rows()).to_csv(base / "manifest.csv", index=False)
    pd.DataFrame(_pairs_rows("train", 1, 2), columns=CANONICAL_PAIR_COLUMNS).to_csv(
        pairs_dir / "pairs_train.csv", index=False
    )
    pd.DataFrame(_pairs_rows("val", 3, 4), columns=CANONICAL_PAIR_COLUMNS).to_csv(
        pairs_dir / "pairs_val.csv", index=False
    )
    pd.DataFrame(_pairs_rows("test", 5, 6), columns=CANONICAL_PAIR_COLUMNS).to_csv(
        pairs_dir / "pairs_test.csv", index=False
    )

    split_meta = build_split_subjects_metadata(
        splits={"train": [1, 2], "val": [3, 4], "test": [5, 6]},
        seed=42,
        neg_per_pos=3,
        impostors_per_pos=3,
        same_finger_policy=True,
        negative_pair_policy="same_finger_other_subject_same_split",
        positive_pair_policy="same_subject_same_finger_plain_to_roll",
        finger_col="frgp",
        resolved_data_dir=root / "data" / "raw" / DATASET_NAME,
        manifest_path=base / "manifest.csv",
    )
    _write_json(pairs_dir / "split_subjects.json", split_meta)

    build_meta = build_pairs_split_build_meta(
        dataset=DATASET_NAME,
        seed=42,
        neg_per_pos=3,
        impostors_per_pos=3,
        finger_col="frgp",
        positive_pair_policy="same_subject_same_finger_plain_to_roll",
        negative_pair_policy="same_finger_other_subject_same_split",
    )
    _write_json(base / "pairs_split_build.meta.json", build_meta)
    return root



def _run_validator(repo_root: Path, *, env_override: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_override is None:
        env["FPRJ_ROOT"] = str(repo_root)
    else:
        env["FPRJ_ROOT"] = env_override
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=repo_root.parent,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )



def test_valid_bundle_passes(tmp_path: Path) -> None:
    repo_root = _write_valid_repo(tmp_path / "repo")
    result = _run_validator(repo_root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SUMMARY: PASS" in result.stdout
    assert "VALIDATION PASSED" in result.stdout



def test_missing_pair_column_fails(tmp_path: Path) -> None:
    repo_root = _write_valid_repo(tmp_path / "repo")
    pairs_path = repo_root / "data" / "manifests" / DATASET_NAME / "pairs" / "pairs_train.csv"
    df = pd.read_csv(pairs_path)
    df = df.drop(columns=["subject_b"])
    df.to_csv(pairs_path, index=False)

    result = _run_validator(repo_root)
    assert result.returncode == 1
    assert "missing canonical pair columns" in result.stdout
    assert "SUMMARY: FAILED" in result.stdout



def test_missing_split_subjects_field_fails(tmp_path: Path) -> None:
    repo_root = _write_valid_repo(tmp_path / "repo")
    split_meta_path = repo_root / "data" / "manifests" / DATASET_NAME / "pairs" / "split_subjects.json"
    payload = json.loads(split_meta_path.read_text(encoding="utf-8"))
    payload.pop("pair_schema_version")
    _write_json(split_meta_path, payload)

    result = _run_validator(repo_root)
    assert result.returncode == 1
    assert "split_subjects.json" in result.stdout
    assert "missing required fields" in result.stdout



def test_missing_pairs_split_build_meta_field_fails(tmp_path: Path) -> None:
    repo_root = _write_valid_repo(tmp_path / "repo")
    build_meta_path = repo_root / "data" / "manifests" / DATASET_NAME / "pairs_split_build.meta.json"
    payload = json.loads(build_meta_path.read_text(encoding="utf-8"))
    payload.pop("pair_schema_version")
    _write_json(build_meta_path, payload)

    result = _run_validator(repo_root)
    assert result.returncode == 1
    assert "pairs_split_build.meta.json" in result.stdout
    assert "missing required fields" in result.stdout



def test_invalid_root_resolution_fails_clearly(tmp_path: Path) -> None:
    result = _run_validator(tmp_path / "unused", env_override=str(tmp_path / "does-not-exist"))
    assert result.returncode == 2
    assert "Could not resolve project root" in result.stderr
