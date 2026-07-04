from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.build_fusion_failure_taxonomy_v1 import (
    METHOD_NAME,
    assert_counts_match_statistical_comparison,
    run_taxonomy,
)


def _write_image(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((48, 56), shade, dtype=np.uint8)
    cv2.line(image, (6, 8), (48, 38), 35, 2)
    assert cv2.imwrite(str(path), image)


def _pair_paths(root: Path, split: str, idx: int) -> tuple[str, str]:
    plain = root / "images" / split / f"plain_{idx}.png"
    roll = root / "images" / split / f"roll_{idx}.png"
    _write_image(plain, 220 - (idx % 6) * 16)
    _write_image(roll, 214 - (idx % 6) * 16)
    return str(plain), str(roll)


def _row(
    root: Path,
    *,
    split: str,
    pair_id: int,
    label: int,
    sourceafis_score: float,
    sift_score: float,
    fusion_score: float,
) -> dict[str, Any]:
    path_a, path_b = _pair_paths(root, split, pair_id)
    subject_a = f"s{pair_id:03d}"
    subject_b = subject_a if label == 1 else f"other{pair_id:03d}"
    return {
        "method": METHOD_NAME,
        "dataset": "toy",
        "split": split,
        "pair_id": str(pair_id),
        "label": int(label),
        "subject_a": subject_a,
        "subject_b": subject_b,
        "finger_position": "3",
        "frgp": "3",
        "path_a": path_a,
        "path_b": path_b,
        "score": float(fusion_score),
        "sourceafis_score": float(sourceafis_score),
        "sift_plain_roll_v2_score": float(sift_score),
        "score_semantics": "toy_probability",
        "higher_is_more_similar": True,
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    fusion_dir = tmp_path / "fusion_full"
    sift_dir = tmp_path / "sift"
    score_dir = fusion_dir / "scores"
    score_dir.mkdir(parents=True)
    sift_dir.mkdir(parents=True)

    val_rows = [
        _row(tmp_path, split="val", pair_id=0, label=0, sourceafis_score=0.2, sift_score=0.2, fusion_score=0.1),
        _row(tmp_path, split="val", pair_id=1, label=0, sourceafis_score=0.8, sift_score=0.6, fusion_score=0.7),
        _row(tmp_path, split="val", pair_id=2, label=1, sourceafis_score=0.9, sift_score=1.0, fusion_score=0.9),
        _row(tmp_path, split="val", pair_id=3, label=1, sourceafis_score=1.0, sift_score=1.1, fusion_score=1.0),
    ]
    test_rows = [
        _row(tmp_path, split="test", pair_id=0, label=1, sourceafis_score=0.1, sift_score=1.2, fusion_score=0.9),
        _row(tmp_path, split="test", pair_id=1, label=1, sourceafis_score=0.9, sift_score=0.4, fusion_score=0.2),
        _row(tmp_path, split="test", pair_id=2, label=1, sourceafis_score=0.9, sift_score=1.1, fusion_score=0.9),
        _row(tmp_path, split="test", pair_id=3, label=1, sourceafis_score=0.1, sift_score=0.3, fusion_score=0.2),
        _row(tmp_path, split="test", pair_id=4, label=0, sourceafis_score=0.9, sift_score=0.3, fusion_score=0.2),
        _row(tmp_path, split="test", pair_id=5, label=0, sourceafis_score=0.1, sift_score=1.2, fusion_score=0.9),
        _row(tmp_path, split="test", pair_id=6, label=0, sourceafis_score=0.1, sift_score=0.3, fusion_score=0.2),
        _row(tmp_path, split="test", pair_id=7, label=0, sourceafis_score=0.9, sift_score=1.1, fusion_score=0.9),
    ]
    for split, rows in (("val", val_rows), ("test", test_rows)):
        pd.DataFrame(rows).to_csv(score_dir / f"scores_toy_{METHOD_NAME}_{split}.csv", index=False)
        sift_rows = []
        for row in rows:
            sift_rows.append(
                {
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "pair_id": row["pair_id"],
                    "label": row["label"],
                    "path_a": row["path_a"],
                    "path_b": row["path_b"],
                    "score": row["sift_plain_roll_v2_score"],
                    "inliers": int(row["sift_plain_roll_v2_score"] * 10),
                    "matches": int(row["sift_plain_roll_v2_score"] * 20) + 1,
                    "k1": 100,
                    "k2": 110,
                    "pair_total_ms": 1.0,
                }
            )
        pd.DataFrame(sift_rows).to_csv(sift_dir / f"scores_toy_sift_plain_roll_v2_{split}.csv", index=False)

    comparison = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "split": "test",
                "target_far": 0.5,
                "rescued_positives": 1,
                "lost_positives": 1,
                "fixed_false_accepts": 1,
                "new_false_accepts": 1,
                "fusion_ta": 2,
                "fusion_fa": 2,
                "sourceafis_ta": 2,
                "sourceafis_fa": 2,
            }
        ]
    )
    comparison_path = fusion_dir / "plain_roll_final_statistical_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    return fusion_dir, sift_dir, comparison_path


def test_fusion_failure_taxonomy_outputs_and_matches_statistical_comparison(tmp_path: Path) -> None:
    fusion_dir, sift_dir, comparison_path = _write_inputs(tmp_path)
    outdir = tmp_path / "taxonomy"

    paths = run_taxonomy(
        datasets=("toy",),
        splits=("val", "test"),
        fusion_dir=fusion_dir,
        sift_score_dir=sift_dir,
        outdir=outdir,
        comparison_csv=comparison_path,
        target_far=0.5,
        repo_root=tmp_path,
    )

    expected_names = {
        "failure_taxonomy_pairs",
        "failure_taxonomy_summary",
        "failure_taxonomy_by_dataset",
        "failure_taxonomy_by_finger",
        "failure_taxonomy_by_score_band",
        "failure_taxonomy_by_quality_band",
        "rescued_positive_examples",
        "lost_positive_examples",
        "fixed_false_accept_examples",
        "new_false_accept_examples",
    }
    assert expected_names.issubset(paths)
    for name in expected_names:
        assert paths[name].exists()

    pairs = pd.read_csv(paths["failure_taxonomy_pairs"])
    test = pairs[pairs["split"] == "test"]
    counts = test["category"].value_counts().to_dict()

    assert counts["rescued_positive"] == 1
    assert counts["lost_positive"] == 1
    assert counts["fixed_false_accept"] == 1
    assert counts["new_false_accept"] == 1
    assert counts["both_correct"] == 2
    assert counts["both_wrong"] == 2
    assert set(test["correctness_category"]) == {
        "both_correct",
        "both_wrong",
        "fusion_only_correct",
        "sourceafis_only_correct",
    }
    assert "a_contrast_proxy" in pairs.columns
    assert "b_sharpness_laplacian_var" in pairs.columns
    assert "sift_score_percentile" in pairs.columns

    by_score = pd.read_csv(paths["failure_taxonomy_by_score_band"])
    assert "sift_high_sourceafis_low" in set(by_score["score_band"])
    assert "sourceafis_high_fusion_suppressed_false_accept" in set(by_score["score_band"])
    assert "Recommendations" in paths["failure_taxonomy_summary"].read_text(encoding="utf-8")

    assert_counts_match_statistical_comparison(
        pairs,
        pd.read_csv(comparison_path),
        target_far=0.5,
    )


def test_fusion_failure_taxonomy_rejects_statistical_comparison_mismatch(tmp_path: Path) -> None:
    fusion_dir, sift_dir, comparison_path = _write_inputs(tmp_path)
    outdir = tmp_path / "taxonomy"
    paths = run_taxonomy(
        datasets=("toy",),
        splits=("val", "test"),
        fusion_dir=fusion_dir,
        sift_score_dir=sift_dir,
        outdir=outdir,
        comparison_csv=comparison_path,
        target_far=0.5,
        repo_root=tmp_path,
    )
    pairs = pd.read_csv(paths["failure_taxonomy_pairs"])
    comparison = pd.read_csv(comparison_path)
    comparison.loc[0, "rescued_positives"] = 99

    with pytest.raises(AssertionError, match="counts do not match"):
        assert_counts_match_statistical_comparison(pairs, comparison, target_far=0.5)
