from __future__ import annotations

import math

import pandas as pd
import pytest

from src.fpbench.universal import calibration


def _feature_rows(split: str = "train") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "toy",
                "split": split,
                "label": 0,
                "finger_position": "1",
                "frgp": "1",
                "sourceafis_score": 0.1,
                "sift_plain_roll_v2_score": 0.2,
                "ppi": 1000,
                "a_mean_intensity": 100,
                "b_mean_intensity": 105,
                "pair_mean_intensity_abs_delta": 5,
            },
            {
                "dataset": "toy",
                "split": split,
                "label": 1,
                "finger_position": "1",
                "frgp": "1",
                "sourceafis_score": 3.0,
                "sift_plain_roll_v2_score": 5.0,
                "ppi": 1000,
                "a_mean_intensity": 120,
                "b_mean_intensity": 118,
                "pair_mean_intensity_abs_delta": 2,
            },
        ]
    )


def test_model_fit_rejects_val_or_test_rows() -> None:
    table = pd.concat([_feature_rows("train"), _feature_rows("val")], ignore_index=True)

    with pytest.raises(calibration.FusionCalibrationError, match="train rows only"):
        calibration.fit_fusion_model(table)


def test_model_fit_accepts_train_rows_only() -> None:
    table = pd.concat([_feature_rows("train"), _feature_rows("train")], ignore_index=True)
    table.loc[2, "label"] = 0
    table.loc[3, "label"] = 1

    model, schema = calibration.fit_fusion_model(table)
    scores = calibration.predict_fusion_scores(model, schema, table)

    assert len(scores) == len(table)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert schema["method"] == "sourceafis_sift_quality_fusion_v1"


def test_threshold_selection_uses_val_negatives_only() -> None:
    scores = pd.DataFrame(
        [
            {"dataset": "toy", "split": "val", "label": 0, "score": 0.2},
            {"dataset": "toy", "split": "val", "label": 0, "score": 0.8},
            {"dataset": "toy", "split": "val", "label": 1, "score": 0.95},
            {"dataset": "toy", "split": "test", "label": 0, "score": 0.99},
        ]
    )

    thresholds = calibration.build_threshold_table_from_scores(scores, target_fars=(0.0, 0.5))
    zero_far = thresholds[thresholds["target_far"] == 0.0].iloc[0]
    half_far = thresholds[thresholds["target_far"] == 0.5].iloc[0]

    assert zero_far["threshold"] > 0.8
    assert zero_far["threshold"] < 0.95
    assert zero_far["calibration_false_accepts"] == 0
    assert half_far["threshold"] == pytest.approx(0.8)
    assert half_far["calibration_false_accepts"] == 1
    assert half_far["calibration_far"] == pytest.approx(0.5)
    assert math.isfinite(float(half_far["threshold"]))
