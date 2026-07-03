from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipelines.benchmark.benchmark_validation_utils import (
    BENCHMARK_CONFIG_SCHEMA_VERSION,
    BENCHMARK_RUN_META_SCHEMA_VERSION,
    validate_run_meta,
)


def test_deep_run_meta_schema_is_validator_compatible(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    summary = tmp_path / "summary.csv"
    run_meta = tmp_path / "run.meta.json"
    pd.DataFrame([{"label": 1, "score": 0.9}, {"label": 0, "score": 0.1}]).to_csv(scores, index=False)
    config = {
        "schema_version": BENCHMARK_CONFIG_SCHEMA_VERSION,
        "method": "deep_pair_reranker_v1",
        "split": "val",
        "pairs_path": str(tmp_path / "pairs.csv"),
        "manifest_path": str(tmp_path / "manifest.csv"),
        "resolved_data_dir": str(tmp_path),
    }
    row = {
        "method": "deep_pair_reranker_v1",
        "split": "val",
        "n_pairs": 2,
        "auc": 1.0,
        "eer": 0.0,
        "tar_at_far_1e_2": 1.0,
        "tar_at_far_1e_3": 1.0,
        "scores_csv": str(scores),
        "config_json": json.dumps(config, ensure_ascii=False),
    }
    pd.DataFrame([row]).to_csv(summary, index=False)
    payload = {
        "schema_version": BENCHMARK_RUN_META_SCHEMA_VERSION,
        "row": row,
        "scores_csv": str(scores),
        "roc_png": str(tmp_path / "roc.png"),
        "summary_csv": str(summary),
        "resolved_data_dir": str(tmp_path),
        "manifest_path": str(tmp_path / "manifest.csv"),
        "pairs_path": str(tmp_path / "pairs.csv"),
        "config": config,
    }
    run_meta.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded = validate_run_meta(
        run_meta,
        expected_row=row,
        expected_scores_csv=scores,
        expected_summary_csv=summary,
        expected_method="deep_pair_reranker_v1",
        expected_split="val",
    )
    assert loaded["schema_version"] == BENCHMARK_RUN_META_SCHEMA_VERSION
