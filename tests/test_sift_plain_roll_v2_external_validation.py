from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.diagnostics.run_sift_plain_roll_v2_external_validation import (
    ScoreRun,
    build_validation_tables,
)


def _write_scores(path: Path, *, split: str, rows: list[tuple[int, float, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "label": label,
                "split": split,
                "path_a": f"C:/fingerprint-research/data/raw/plain_{idx}.png",
                "path_b": f"C:/fingerprint-research/data/raw/roll_{idx}.png",
                "score": score,
                "inliers": inliers,
                "matches": matches,
                "k1": 100,
                "k2": 100,
            }
            for idx, (label, score, inliers, matches) in enumerate(rows)
        ]
    ).to_csv(path, index=False)


def _run(tmp_path: Path, dataset: str, method: str, split: str) -> ScoreRun:
    score_csv = tmp_path / dataset / f"scores_{method}_{split}.csv"
    return ScoreRun(
        dataset=dataset,
        dataset_dir=tmp_path / dataset,
        split=split,
        pairs_csv=tmp_path / dataset / f"pairs_{split}.csv",
        method=method,
        score_csv=score_csv,
        roc_png=tmp_path / dataset / f"roc_{method}_{split}.png",
        run_meta_json=tmp_path / dataset / f"run_{method}_{split}.meta.json",
        command=["python", "evaluate.py"],
    )


def test_external_validation_calibrates_on_val_negatives_and_reports_test(tmp_path: Path) -> None:
    val_rows = [(0, 0.10, 1, 5), (0, 0.20, 2, 5), (0, 0.80, 8, 10), (1, 0.30, 3, 6), (1, 0.90, 9, 12)]
    test_rows = [(0, 0.85, 8, 10), (0, 0.10, 1, 5), (1, 0.70, 7, 10), (1, 0.95, 9, 12)]

    runs = []
    for method in ("sift", "sift_plain_roll_v2"):
        val_run = _run(tmp_path, "toy", method, "val")
        test_run = _run(tmp_path, "toy", method, "test")
        _write_scores(val_run.score_csv, split="val", rows=val_rows)
        _write_scores(test_run.score_csv, split="test", rows=test_rows)
        runs.extend([val_run, test_run])

    thresholds, metrics, false_accepts, false_rejects = build_validation_tables(
        runs,
        target_fars=(0.34,),
        top_n_cases=5,
    )

    sift_threshold = thresholds[
        (thresholds["method"] == "sift")
        & (thresholds["variant"] == "current_score")
        & (thresholds["dataset"] == "toy")
    ].iloc[0]
    assert sift_threshold["threshold"] == 0.80
    assert sift_threshold["calibration_false_accepts"] == 1

    test_metric = metrics[
        (metrics["split"] == "test")
        & (metrics["method"] == "sift")
        & (metrics["variant"] == "current_score")
    ].iloc[0]
    assert test_metric["ta"] == 1
    assert test_metric["fr"] == 1
    assert test_metric["fa"] == 1
    assert test_metric["tr"] == 1
    assert test_metric["tar"] == 0.5
    assert test_metric["far"] == 0.5

    assert {"current_score", "inliers"} <= set(metrics[metrics["method"] == "sift"]["variant"])
    assert set(metrics[metrics["method"] == "sift_plain_roll_v2"]["variant"]) == {"official_score"}
    assert len(false_accepts) == 3
    assert len(false_rejects) == 3
