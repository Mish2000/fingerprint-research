from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.diagnostics.run_sift_plain_roll_v2_grid3_validation import (
    GRID3_FAMILY,
    GRID3_NAME,
    GUARDRAIL_BY_NAME,
    apply_guardrail,
    assert_grid3_output_coverage,
    build_grid3_decisions,
    build_visual_audit_case_index,
    build_winning_crop_analysis,
    evaluate_guardrail_candidates_val,
    select_guardrail_from_val,
)


def _path(dataset: str, idx: int, capture: str = "plain", frgp: int = 5) -> str:
    return f"C:/fingerprint-research/data/raw/NIST/{dataset}/images/1000/png/{capture}/{idx:08d}_{capture}_1000_{frgp:02d}.png"


def _source_row(
    dataset: str,
    split: str,
    label: int,
    idx: int,
    *,
    v2_score: float,
    v2_inliers: int = 4,
    frgp: int = 5,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "split": split,
        "label": label,
        "path_a": _path(dataset, idx, "plain", frgp),
        "path_b": _path(dataset, idx + 1000, "roll", frgp),
        "subject_a": f"{idx:04d}",
        "subject_b": f"{idx:04d}" if label else f"{idx + 1000:04d}",
        "frgp_a": frgp,
        "frgp_b": frgp,
        "frgp": frgp,
        "v2_official_score": v2_score,
        "v2_inliers": v2_inliers,
        "v2_matches": v2_inliers + 5,
        "v2_k1": 100,
        "v2_k2": 100,
    }


def _grid3_row(
    dataset: str,
    split: str,
    label: int,
    idx: int,
    *,
    grid3_score: float,
    grid3_inliers: int = 8,
    crop_index: int = 1,
    frgp: int = 5,
) -> dict[str, object]:
    crop_label = ("left_60", "center_60", "right_60")[crop_index]
    crop_bbox = [10 + crop_index, 20, 300 + crop_index, 700]
    roi_bbox = [0, 20, 768, 700]
    diagnostic = {
        "winning_crop_index": crop_index,
        "winning_crop_label": crop_label,
        "winning_crop_bbox": crop_bbox,
        "roll_roi_bbox": roi_bbox,
        "per_crop_scores": [
            {"crop_index": 0, "crop_label": "left_60", "score": 1.0, "matches": 5, "inliers": 2, "crop_bbox": [0, 20, 400, 700]},
            {"crop_index": 1, "crop_label": "center_60", "score": 2.0, "matches": 6, "inliers": 3, "crop_bbox": [100, 20, 500, 700]},
            {"crop_index": 2, "crop_label": "right_60", "score": 3.0, "matches": 7, "inliers": 4, "crop_bbox": [200, 20, 600, 700]},
        ],
    }
    return {
        "dataset": dataset,
        "split": split,
        "label": label,
        "path_a": _path(dataset, idx, "plain", frgp),
        "path_b": _path(dataset, idx + 1000, "roll", frgp),
        "frgp": frgp,
        "candidate_name": GRID3_NAME,
        "candidate_family": GRID3_FAMILY,
        "probe_kind": "image_scalar",
        "grid3_score": grid3_score,
        "grid3_matches": grid3_inliers + 4,
        "grid3_inliers": grid3_inliers,
        "grid3_k1": 100,
        "grid3_k2": 100,
        "diagnostic_json": json.dumps(diagnostic),
        "winning_crop_index": crop_index,
        "winning_crop_label": crop_label,
        "winning_crop_bbox": json.dumps(crop_bbox),
        "roll_roi_bbox": json.dumps(roi_bbox),
        "crop_geometry": f"{crop_label} bbox={crop_bbox} roi={roi_bbox}",
        "winning_crop_position": crop_label.split("_")[0],
    }


def _thresholds(dataset: str = "nist_sd300b") -> tuple[pd.DataFrame, pd.DataFrame]:
    grid3 = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "candidate_name": GRID3_NAME,
                "candidate_family": GRID3_FAMILY,
                "target_far": 0.01,
                "grid3_threshold": 5.0,
                "calibration_split": "val",
                "selected_by_val": True,
            }
        ]
    )
    v2 = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "target_far": 0.01,
                "v2_threshold": 5.0,
                "calibration_split": "val",
            }
        ]
    )
    return grid3, v2


def _decision_row(
    split: str,
    label: int,
    *,
    v2_accepted: bool,
    grid3_accepted: bool,
    grid3_inliers: int,
    v2_score: float,
    grid3_score: float,
    v2_inliers: int = 4,
) -> dict[str, object]:
    grid3_threshold = 5.0
    return {
        "dataset": "nist_sd300b",
        "split": split,
        "target_far": 0.01,
        "label": label,
        "v2_accepted": v2_accepted,
        "grid3_accepted": grid3_accepted,
        "v2_score": v2_score,
        "grid3_score": grid3_score,
        "score_delta_grid3_minus_v2": grid3_score - v2_score,
        "v2_threshold": 5.0,
        "grid3_threshold": grid3_threshold,
        "grid3_score_margin_ratio": (grid3_score - grid3_threshold) / grid3_threshold,
        "v2_inliers": v2_inliers,
        "grid3_inliers": grid3_inliers,
        "winning_crop_index": 1,
        "winning_crop_label": "center_60",
        "winning_crop_bbox": "[100,20,500,700]",
        "roll_roi_bbox": "[0,20,768,700]",
        "crop_geometry": "center_60 bbox=[100,20,500,700] roi=[0,20,768,700]",
    }


def _guardrail_fixture() -> pd.DataFrame:
    rows = [
        _decision_row("val", 1, v2_accepted=False, grid3_accepted=True, grid3_inliers=8, v2_score=0.0, grid3_score=7.0, v2_inliers=0),
        _decision_row("val", 1, v2_accepted=True, grid3_accepted=True, grid3_inliers=8, v2_score=6.0, grid3_score=7.0),
        _decision_row("val", 0, v2_accepted=False, grid3_accepted=True, grid3_inliers=4, v2_score=0.0, grid3_score=10.0, v2_inliers=0),
        _decision_row("val", 0, v2_accepted=False, grid3_accepted=True, grid3_inliers=4, v2_score=0.0, grid3_score=9.0, v2_inliers=0),
        _decision_row("test", 1, v2_accepted=False, grid3_accepted=True, grid3_inliers=2, v2_score=0.0, grid3_score=50.0, v2_inliers=0),
        _decision_row("test", 0, v2_accepted=False, grid3_accepted=True, grid3_inliers=2, v2_score=0.0, grid3_score=50.0, v2_inliers=0),
    ]
    return pd.DataFrame(rows)


def test_grid3_comparison_uses_val_for_guardrail_selection() -> None:
    decisions = _guardrail_fixture()

    val_candidates = evaluate_guardrail_candidates_val(decisions)
    selected = select_guardrail_from_val(val_candidates)

    assert selected == "winning_crop_inliers_ge_8"
    assert set(val_candidates["split"]) == {"val"}


def test_test_is_not_used_for_guardrail_selection() -> None:
    decisions = _guardrail_fixture()
    selected_before = select_guardrail_from_val(evaluate_guardrail_candidates_val(decisions))
    mutated = decisions.copy()
    mutated.loc[mutated["split"] == "test", "grid3_inliers"] = 99
    mutated.loc[mutated["split"] == "test", "v2_score"] = 99.0
    mutated.loc[mutated["split"] == "test", "label"] = 1 - mutated.loc[mutated["split"] == "test", "label"]

    selected_after = select_guardrail_from_val(evaluate_guardrail_candidates_val(mutated))

    assert selected_after == selected_before


def test_winning_crop_analysis_preserves_pair_alignment() -> None:
    source = pd.DataFrame(
        [
            _source_row("nist_sd300b", "test", 1, 1, v2_score=1.0),
            _source_row("nist_sd300b", "test", 0, 2, v2_score=1.0),
        ]
    )
    grid3 = pd.DataFrame(
        [
            _grid3_row("nist_sd300b", "test", 0, 2, grid3_score=7.0, crop_index=0),
            _grid3_row("nist_sd300b", "test", 1, 1, grid3_score=8.0, crop_index=2),
        ]
    )
    grid3_thresholds, v2_thresholds = _thresholds()

    decisions = build_grid3_decisions(source, grid3, grid3_thresholds, v2_thresholds, target_fars=(0.01,))
    analysis = build_winning_crop_analysis(decisions)
    by_path = {row["path_a"]: int(row["winning_crop_index"]) for _, row in analysis.iterrows()}

    assert by_path[_path("nist_sd300b", 1, "plain", 5)] == 2
    assert by_path[_path("nist_sd300b", 2, "plain", 5)] == 0


def test_visual_audit_rows_include_crop_index_and_geometry() -> None:
    row = _decision_row("test", 0, v2_accepted=False, grid3_accepted=True, grid3_inliers=8, v2_score=1.0, grid3_score=9.0)
    row.update(
        {
            "frgp": 5,
            "path_a": _path("nist_sd300b", 1, "plain", 5),
            "path_b": _path("nist_sd300b", 1001, "roll", 5),
            "decision_category": "new_false_accept",
            "v2_failure_severity": "",
            "grid3_matches": 12,
            "near_threshold_grid3_accept": False,
            "high_confidence_grid3_accept": True,
        }
    )

    index = build_visual_audit_case_index(pd.DataFrame([row]), top_n=1)

    assert not index.empty
    assert int(index.iloc[0]["winning_crop_index"]) == 1
    assert "bbox=" in str(index.iloc[0]["crop_geometry"])


def test_guardrail_does_not_modify_original_grid3_scores_only_decisions() -> None:
    decisions = pd.DataFrame(
        [
            _decision_row(
                "val",
                0,
                v2_accepted=False,
                grid3_accepted=True,
                grid3_inliers=2,
                v2_score=1.0,
                grid3_score=9.0,
            )
        ]
    )

    guarded = apply_guardrail(decisions, GUARDRAIL_BY_NAME["winning_crop_inliers_ge_8"])

    assert guarded.loc[0, "grid3_score"] == pytest.approx(decisions.loc[0, "grid3_score"])
    assert bool(guarded.loc[0, "grid3_accepted"]) is True
    assert bool(guarded.loc[0, "grid3_guarded_accepted"]) is False
    assert bool(guarded.loc[0, "guardrail_modified_original_scores"]) is False


def test_outputs_include_both_sd300b_and_sd300c() -> None:
    metrics = pd.DataFrame(
        [
            {"dataset": dataset, "split": split, "target_far": 0.01}
            for dataset in ("nist_sd300b", "nist_sd300c")
            for split in ("val", "test")
        ]
    )

    assert_grid3_output_coverage(metrics)

    with pytest.raises(AssertionError):
        assert_grid3_output_coverage(metrics[metrics["dataset"] == "nist_sd300b"])
