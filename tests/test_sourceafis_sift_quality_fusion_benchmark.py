from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pipelines.benchmark import run_plain_roll_final_benchmark as final
from pipelines.benchmark.run_sourceafis_sift_quality_fusion_benchmark import run_benchmark
from pipelines.benchmark.run_sourceafis_sift_quality_fusion_phase15 import ABLATION_VARIANTS, run_ablation_validation
from pipelines.benchmark.train_sourceafis_sift_quality_fusion import train_fusion


def _write_image(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((40, 48), shade, dtype=np.uint8)
    cv2.line(image, (5, 5), (40, 30), 40, 2)
    assert cv2.imwrite(str(path), image)


def _pairs(repo: Path, split: str, scores: list[tuple[int, float, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (label, _source_score, _sift_score) in enumerate(scores):
        plain = repo / "images" / split / f"plain_{idx}.png"
        roll = repo / "images" / split / f"roll_{idx}.png"
        _write_image(plain, 220 - idx * 15)
        _write_image(roll, 215 - idx * 15)
        subject_a = f"s{idx}"
        rows.append(
            {
                "pair_id": str(idx),
                "label": int(label),
                "split": split,
                "subject_a": subject_a,
                "subject_b": subject_a if int(label) == 1 else f"other-{idx}",
                "frgp": "2",
                "path_a": str(plain),
                "path_b": str(roll),
            }
        )
    return rows


def _write_score_files(
    *,
    rows: list[dict[str, Any]],
    values: list[tuple[int, float, float]],
    sourceafis_path: Path,
    sift_path: Path,
) -> None:
    sourceafis_rows = []
    sift_rows = []
    for row, (_label, source_score, sift_score) in zip(rows, values):
        common = {
            "dataset": "toy",
            "split": row["split"],
            "pair_id": row["pair_id"],
            "label": row["label"],
            "subject_a": row["subject_a"],
            "subject_b": row["subject_b"],
            "finger_position": row["frgp"],
            "frgp": row["frgp"],
            "path_a": row["path_a"],
            "path_b": row["path_b"],
            "dpi_a": 1000,
            "dpi_b": 1000,
            "pair_total_ms": 1.0,
        }
        sourceafis_rows.append({**common, "method": "sourceafis_open", "score": source_score, "raw_score": source_score})
        sift_rows.append(
            {
                **common,
                "score": sift_score,
                "inliers": int(sift_score * 10),
                "matches": int(sift_score * 20) + 1,
                "k1": 100,
                "k2": 100,
            }
        )
    sourceafis_path.parent.mkdir(parents=True, exist_ok=True)
    sift_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sourceafis_rows).to_csv(sourceafis_path, index=False)
    pd.DataFrame(sift_rows).to_csv(sift_path, index=False)


def test_fusion_benchmark_smoke_outputs_part1_compatible_schema(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "data" / "manifests" / "toy"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.csv").write_text("path\n", encoding="utf-8")

    train_values = [(0, 0.1, 0.5), (0, 0.2, 0.4), (1, 2.5, 4.0), (1, 3.0, 4.5)]
    val_values = [(0, 0.2, 0.7), (0, 0.8, 1.2), (1, 2.6, 4.1), (1, 2.9, 4.4)]
    test_values = [(0, 0.3, 0.8), (0, 1.0, 1.5), (1, 2.2, 3.8), (1, 3.1, 4.7)]
    train_rows = _pairs(tmp_path, "train", train_values)
    val_rows = _pairs(tmp_path, "val", val_values)
    test_rows = _pairs(tmp_path, "test", test_values)
    pd.DataFrame(train_rows).to_csv(manifest_dir / "pairs_train.csv", index=False)
    pd.DataFrame(val_rows).to_csv(manifest_dir / "pairs_val.csv", index=False)
    pd.DataFrame(test_rows).to_csv(manifest_dir / "pairs_test.csv", index=False)

    selected_dir = tmp_path / "selected_pairs"
    selected_dir.mkdir()
    pd.DataFrame(val_rows).to_csv(selected_dir / "pairs_toy_val.csv", index=False)
    pd.DataFrame(test_rows).to_csv(selected_dir / "pairs_toy_test.csv", index=False)

    train_score_dir = tmp_path / "train_scores"
    source_score_dir = tmp_path / "sourceafis_scores"
    sift_score_dir = tmp_path / "sift_scores"
    _write_score_files(
        rows=train_rows,
        values=train_values,
        sourceafis_path=train_score_dir / "scores_toy_sourceafis_open_train.csv",
        sift_path=train_score_dir / "scores_toy_sift_plain_roll_v2_train.csv",
    )
    for split, rows, values in (("val", val_rows, val_values), ("test", test_rows, test_values)):
        _write_score_files(
            rows=rows,
            values=values,
            sourceafis_path=source_score_dir / f"scores_toy_sourceafis_open_{split}.csv",
            sift_path=sift_score_dir / f"scores_toy_sift_plain_roll_v2_{split}.csv",
        )

    train_paths = train_fusion(
        datasets=("toy",),
        train_score_dir=train_score_dir,
        outdir=tmp_path / "fusion",
        repo_root=tmp_path,
        save_training_features=True,
    )
    paths = run_benchmark(
        datasets=("toy",),
        splits=("val", "test"),
        outdir=tmp_path / "fusion_benchmark",
        model_dir=Path(train_paths["model"]).parent,
        selected_pairs_dir=selected_dir,
        sourceafis_score_dir=source_score_dir,
        sift_plain_roll_score_dir=sift_score_dir,
        target_fars=(0.5,),
        repo_root=tmp_path,
    )

    metrics = pd.read_csv(paths["metrics"])
    thresholds = pd.read_csv(paths["thresholds"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    fusion_scores = pd.read_csv(tmp_path / "fusion_benchmark" / "scores" / "scores_toy_sourceafis_sift_quality_fusion_v1_test.csv")

    assert list(metrics.columns) == final.METRICS_COLUMNS
    assert list(thresholds.columns) == final.THRESHOLD_COLUMNS
    assert "test" in set(metrics["split"])
    assert thresholds.iloc[0]["selection_rule"] == "lowest VAL negative-score threshold with VAL FAR <= target"
    assert fusion_scores["score"].between(0, 1).all()
    assert paths["threshold_sweep"].exists()
    assert paths["tar_far_distribution"].exists()
    assert paths["failures"].exists()
    assert manifest["schema_version"] == "plain_roll_final_fusion_benchmark_v1"


def test_phase15_ablation_outputs_all_variants_without_test_leakage(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "data" / "manifests" / "toy"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.csv").write_text("path\n", encoding="utf-8")

    train_values = [(0, 0.1, 0.5), (0, 0.2, 0.4), (1, 2.5, 4.0), (1, 3.0, 4.5)]
    val_values = [(0, 0.2, 0.7), (0, 0.8, 1.2), (1, 2.6, 4.1), (1, 2.9, 4.4)]
    test_values = [(0, 0.3, 0.8), (0, 1.0, 1.5), (1, 2.2, 3.8), (1, 3.1, 4.7)]
    train_rows = _pairs(tmp_path, "train", train_values)
    val_rows = _pairs(tmp_path, "val", val_values)
    test_rows = _pairs(tmp_path, "test", test_values)
    pd.DataFrame(train_rows).to_csv(manifest_dir / "pairs_train.csv", index=False)
    pd.DataFrame(val_rows).to_csv(manifest_dir / "pairs_val.csv", index=False)
    pd.DataFrame(test_rows).to_csv(manifest_dir / "pairs_test.csv", index=False)

    selected_dir = tmp_path / "selected_pairs"
    selected_dir.mkdir()
    pd.DataFrame(val_rows).to_csv(selected_dir / "pairs_toy_val.csv", index=False)
    pd.DataFrame(test_rows).to_csv(selected_dir / "pairs_toy_test.csv", index=False)

    train_score_dir = tmp_path / "train_scores"
    source_score_dir = tmp_path / "sourceafis_scores"
    sift_score_dir = tmp_path / "sift_scores"
    _write_score_files(
        rows=train_rows,
        values=train_values,
        sourceafis_path=train_score_dir / "scores_toy_sourceafis_open_train.csv",
        sift_path=train_score_dir / "scores_toy_sift_plain_roll_v2_train.csv",
    )
    for split, rows, values in (("val", val_rows, val_values), ("test", test_rows, test_values)):
        _write_score_files(
            rows=rows,
            values=values,
            sourceafis_path=source_score_dir / f"scores_toy_sourceafis_open_{split}.csv",
            sift_path=sift_score_dir / f"scores_toy_sift_plain_roll_v2_{split}.csv",
        )

    paths = run_ablation_validation(
        datasets=("toy",),
        splits=("val", "test"),
        train_score_dir=train_score_dir,
        outdir=tmp_path / "ablation",
        selected_pairs_dir=selected_dir,
        sourceafis_score_dir=source_score_dir,
        sift_plain_roll_score_dir=sift_score_dir,
        target_fars=(0.5,),
        repo_root=tmp_path,
    )

    metrics = pd.read_csv(paths["metrics"])
    thresholds = pd.read_csv(paths["thresholds"])
    variants = {variant.name for variant in ABLATION_VARIANTS}

    assert list(metrics.columns) == final.METRICS_COLUMNS
    assert list(thresholds.columns) == final.THRESHOLD_COLUMNS
    assert set(metrics["method"]) == variants
    assert set(thresholds["method"]) == variants
    assert set(metrics["split"]) == {"val", "test"}
    assert paths["summary"].exists()
    for variant in variants:
        manifest = json.loads((tmp_path / "ablation" / "model" / variant / "training_manifest.json").read_text())
        assert manifest["protocol"]["fit_splits"] == ["train"]
        assert manifest["protocol"]["no_test_leakage"] is True

    score_csv = tmp_path / "ablation" / "scores" / "scores_toy_sourceafis_only_calibrated_test.csv"
    scores = pd.read_csv(score_csv)
    assert scores["score"].between(0, 1).all()
    assert scores["pair_total_ms"].isna().all()
    assert set(scores["pair_total_ms_semantics"]) == {
        "not_measured_per_pair; run_level_avg_pair_ms is batch prediction average"
    }
