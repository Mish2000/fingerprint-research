from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.benchmark.benchmark_validation_utils import (
    BENCHMARK_CONFIG_SCHEMA_VERSION,
    BENCHMARK_RUN_META_SCHEMA_VERSION,
    validate_run_meta,
    validate_scores_csv,
)


def _write_scores_csv(path: Path) -> None:
    pd.DataFrame([
        {"label": 1, "score": 0.91},
        {"label": 0, "score": 0.12},
    ]).to_csv(path, index=False)


def test_validate_scores_csv_accepts_canonical_label_score_file(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    _write_scores_csv(scores_csv)
    df = validate_scores_csv(scores_csv, expected_n_pairs=2)
    assert list(df.columns) == ["label", "score"]


def test_validate_scores_csv_rejects_noncanonical_columns(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    pd.DataFrame([{"y_true": 1, "score": 0.91}, {"y_true": 0, "score": 0.12}]).to_csv(scores_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_scores_csv(scores_csv, expected_n_pairs=2)


def test_validate_run_meta_requires_schema_versions_and_config_alignment(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    _write_scores_csv(scores_csv)
    summary_csv = tmp_path / "results_summary.csv"
    config = {
        "schema_version": BENCHMARK_CONFIG_SCHEMA_VERSION,
        "method": "dl_quick",
        "split": "val",
        "pairs_path": "/tmp/pairs_val.csv",
        "manifest_path": "/tmp/manifest.csv",
        "resolved_data_dir": "/tmp",
        "extra": 1,
    }
    row = {
        "method": "dl_quick",
        "split": "val",
        "n_pairs": 2,
        "auc": 1.0,
        "eer": 0.0,
        "tar_at_far_1e_2": 1.0,
        "tar_at_far_1e_3": 1.0,
        "scores_csv": str(scores_csv),
        "config_json": json.dumps(config, sort_keys=True),
    }
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    run_meta = {
        "schema_version": BENCHMARK_RUN_META_SCHEMA_VERSION,
        "row": row,
        "scores_csv": str(scores_csv),
        "roc_png": str(tmp_path / "roc.png"),
        "summary_csv": str(summary_csv),
        "resolved_data_dir": "/tmp",
        "manifest_path": "/tmp/manifest.csv",
        "pairs_path": "/tmp/pairs_val.csv",
        "config": config,
    }
    run_meta_path = tmp_path / "run.meta.json"
    run_meta_path.write_text(json.dumps(run_meta), encoding="utf-8")

    payload = validate_run_meta(
        run_meta_path,
        expected_row=row,
        expected_scores_csv=scores_csv,
        expected_summary_csv=summary_csv,
        expected_method="dl_quick",
        expected_split="val",
    )
    assert payload["schema_version"] == BENCHMARK_RUN_META_SCHEMA_VERSION


def test_validate_run_meta_rejects_missing_config_schema_version(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    _write_scores_csv(scores_csv)
    summary_csv = tmp_path / "results_summary.csv"
    config = {
        "method": "dl_quick",
        "split": "val",
        "pairs_path": "/tmp/pairs_val.csv",
        "manifest_path": "/tmp/manifest.csv",
        "resolved_data_dir": "/tmp",
    }
    row = {
        "method": "dl_quick",
        "split": "val",
        "n_pairs": 2,
        "auc": 1.0,
        "eer": 0.0,
        "tar_at_far_1e_2": 1.0,
        "tar_at_far_1e_3": 1.0,
        "scores_csv": str(scores_csv),
        "config_json": json.dumps(config, sort_keys=True),
    }
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    run_meta = {
        "schema_version": BENCHMARK_RUN_META_SCHEMA_VERSION,
        "row": row,
        "scores_csv": str(scores_csv),
        "roc_png": str(tmp_path / "roc.png"),
        "summary_csv": str(summary_csv),
        "resolved_data_dir": "/tmp",
        "manifest_path": "/tmp/manifest.csv",
        "pairs_path": "/tmp/pairs_val.csv",
        "config": config,
    }
    run_meta_path = tmp_path / "run.meta.json"
    run_meta_path.write_text(json.dumps(run_meta), encoding="utf-8")

    with pytest.raises(ValueError, match="config.schema_version"):
        validate_run_meta(
            run_meta_path,
            expected_row=row,
            expected_scores_csv=scores_csv,
            expected_summary_csv=summary_csv,
            expected_method="dl_quick",
            expected_split="val",
        )
