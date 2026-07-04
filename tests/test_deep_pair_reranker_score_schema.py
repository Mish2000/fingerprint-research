from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines.benchmark.benchmark_validation_utils import validate_scores_csv


def test_deep_score_csv_schema_is_validator_compatible(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    pd.DataFrame([
        {"method": "deep_pair_reranker_v1", "label": 1, "score": 0.8, "pair_id": "p1"},
        {"method": "deep_pair_reranker_v1", "label": 0, "score": 0.2, "pair_id": "p2"},
    ]).to_csv(path, index=False)
    df = validate_scores_csv(path, expected_n_pairs=2)
    assert list(df["label"]) == [1, 0]
