from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest

from src.fpbench.universal.fusion_features import (
    FeatureJoinError,
    build_feature_table,
    default_numeric_feature_columns,
)


def _write_image(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((48, 64), shade, dtype=np.uint8)
    cv2.line(image, (8, 8), (56, 40), 30, 2)
    assert cv2.imwrite(str(path), image)


def _pair_rows(tmp_path: Path, split: str = "train") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, label in enumerate([0, 1]):
        plain = tmp_path / "data" / "raw" / "NIST" / "toy" / "images" / "1000" / "png" / "plain" / f"plain_{split}_{idx}.png"
        roll = tmp_path / "data" / "raw" / "NIST" / "toy" / "images" / "1000" / "png" / "roll" / f"roll_{split}_{idx}.png"
        _write_image(plain, 210 - idx * 20)
        _write_image(roll, 205 - idx * 20)
        subject_a = f"s{idx}"
        rows.append(
            {
                "pair_id": str(idx),
                "label": label,
                "split": split,
                "subject_a": subject_a,
                "subject_b": subject_a if label else f"other-{idx}",
                "frgp": "3",
                "path_a": str(plain),
                "path_b": str(roll),
            }
        )
    return rows


def _write_scores(path: Path, pairs: list[dict[str, Any]], column: str, values: list[float]) -> None:
    rows = []
    for row, value in zip(pairs, values):
        rows.append(
            {
                "dataset": "toy",
                "split": row["split"],
                "pair_id": row["pair_id"],
                "label": row["label"],
                "subject_a": row["subject_a"],
                "subject_b": row["subject_b"],
                "finger_position": row["frgp"],
                "path_a": row["path_a"],
                "path_b": row["path_b"],
                column: value,
                "dpi_a": 1000,
                "dpi_b": 1000,
                "inliers": int(value * 10),
                "matches": int(value * 20) + 1,
                "k1": 100,
                "k2": 110,
                "pair_total_ms": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_feature_table_joins_scores_one_to_one_and_preserves_labels(tmp_path: Path) -> None:
    pairs = _pair_rows(tmp_path)
    pairs_csv = tmp_path / "pairs_train.csv"
    source_scores = tmp_path / "scores_toy_sourceafis_open_train.csv"
    sift_scores = tmp_path / "scores_toy_sift_plain_roll_v2_train.csv"
    pd.DataFrame(pairs).to_csv(pairs_csv, index=False)
    _write_scores(source_scores, pairs, "raw_score", [0.2, 2.5])
    _write_scores(sift_scores, pairs, "score", [1.1, 4.2])

    table = build_feature_table(
        dataset="toy",
        split="train",
        pairs_csv=pairs_csv,
        sourceafis_scores_csv=source_scores,
        sift_plain_roll_scores_csv=sift_scores,
        repo_root=tmp_path,
    )

    assert len(table) == 2
    assert table["pair_id"].astype(str).tolist() == ["0", "1"]
    assert table["label"].astype(int).tolist() == [0, 1]
    assert table["sourceafis_score"].tolist() == [0.2, 2.5]
    assert table["sift_plain_roll_v2_score"].tolist() == [1.1, 4.2]
    assert table["ppi"].tolist() == [1000.0, 1000.0]
    assert "a_sharpness_laplacian_var" in table.columns
    assert "pair_mean_intensity_abs_delta" in table.columns
    assert "pair_id" not in default_numeric_feature_columns(table)


def test_feature_table_rejects_duplicate_score_keys(tmp_path: Path) -> None:
    pairs = _pair_rows(tmp_path)
    pairs_csv = tmp_path / "pairs_train.csv"
    source_scores = tmp_path / "scores_toy_sourceafis_open_train.csv"
    sift_scores = tmp_path / "scores_toy_sift_plain_roll_v2_train.csv"
    pd.DataFrame(pairs).to_csv(pairs_csv, index=False)
    _write_scores(source_scores, pairs + [pairs[0]], "raw_score", [0.2, 2.5, 0.3])
    _write_scores(sift_scores, pairs, "score", [1.1, 4.2])

    with pytest.raises(FeatureJoinError, match="duplicate pair keys"):
        build_feature_table(
            dataset="toy",
            split="train",
            pairs_csv=pairs_csv,
            sourceafis_scores_csv=source_scores,
            sift_plain_roll_scores_csv=sift_scores,
            repo_root=tmp_path,
        )


def test_feature_table_rejects_score_context_mismatch(tmp_path: Path) -> None:
    pairs = _pair_rows(tmp_path)
    pairs_csv = tmp_path / "pairs_train.csv"
    source_scores = tmp_path / "scores_toy_sourceafis_open_train.csv"
    sift_scores = tmp_path / "scores_toy_sift_plain_roll_v2_train.csv"
    pd.DataFrame(pairs).to_csv(pairs_csv, index=False)
    _write_scores(source_scores, pairs, "raw_score", [0.2, 2.5])
    bad_pairs = [dict(row) for row in pairs]
    bad_pairs[0]["path_a"] = str(tmp_path / "wrong_plain.png")
    _write_scores(sift_scores, bad_pairs, "score", [1.1, 4.2])

    with pytest.raises(FeatureJoinError, match="path_a values do not match pair bundle"):
        build_feature_table(
            dataset="toy",
            split="train",
            pairs_csv=pairs_csv,
            sourceafis_scores_csv=source_scores,
            sift_plain_roll_scores_csv=sift_scores,
            repo_root=tmp_path,
        )
