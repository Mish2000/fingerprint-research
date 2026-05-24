from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.diagnostics.run_sift_plain_roll_v2_hypothesis_tests import (
    assert_required_output_coverage,
    assert_research_only_candidate_names,
    assert_v2_scores_unchanged,
    build_candidate_decisions,
    build_candidate_metrics,
    build_fusion_probe_scores,
    calibrate_conservative_or_thresholds,
    calibrate_reference_v2_thresholds,
    calibrate_scalar_candidate_thresholds,
    candidate_registry,
    select_candidates_from_val,
)


def _source_row(
    dataset: str,
    split: str,
    label: int,
    idx: int,
    *,
    canonical: float,
    inliers: float,
    v2: float,
    frgp: int = 5,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "split": split,
        "label": label,
        "path_a": f"C:/fingerprint-research/data/raw/{dataset}/plain_{idx:03d}_1000_{frgp:02d}.png",
        "path_b": f"C:/fingerprint-research/data/raw/{dataset}/roll_{idx:03d}_1000_{frgp:02d}.png",
        "frgp": frgp,
        "subject_a": f"{idx:04d}",
        "subject_b": f"{idx:04d}" if label else f"{idx + 1000:04d}",
        "frgp_a": frgp,
        "frgp_b": frgp,
        "canonical_current_score": canonical,
        "sift_inliers_score": inliers,
        "canonical_inliers": inliers,
        "canonical_matches": inliers + 2,
        "canonical_k1": 100,
        "canonical_k2": 100,
        "v2_official_score": v2,
        "v2_inliers": int(v2),
        "v2_matches": int(v2) + 2,
        "v2_k1": 100,
        "v2_k2": 100,
    }


def _candidate_row(
    dataset: str,
    split: str,
    label: int,
    idx: int,
    score: float,
    *,
    name: str = "research_only::geometry_probe_v1:affine_partial_2d",
    family: str = "geometry_probe_v1",
    frgp: int = 5,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "split": split,
        "label": label,
        "path_a": f"C:/fingerprint-research/data/raw/{dataset}/plain_{idx:03d}_1000_{frgp:02d}.png",
        "path_b": f"C:/fingerprint-research/data/raw/{dataset}/roll_{idx:03d}_1000_{frgp:02d}.png",
        "frgp": frgp,
        "candidate_name": name,
        "candidate_family": family,
        "probe_kind": "image_scalar",
        "score": score,
        "matches": 10,
        "inliers": int(score),
        "k1": 100,
        "k2": 100,
        "diagnostic_json": "{}",
        "research_only": True,
    }


def test_val_only_threshold_calibration_ignores_test_negatives() -> None:
    candidate = "research_only::geometry_probe_v1:affine_partial_2d"
    rows = [
        _candidate_row("toy", "val", 0, 1, 0.10, name=candidate),
        _candidate_row("toy", "val", 0, 2, 0.20, name=candidate),
        _candidate_row("toy", "val", 0, 3, 0.80, name=candidate),
        _candidate_row("toy", "val", 1, 4, 0.90, name=candidate),
        _candidate_row("toy", "test", 0, 5, 99.0, name=candidate),
        _candidate_row("toy", "test", 1, 6, 0.70, name=candidate),
    ]
    thresholds = calibrate_scalar_candidate_thresholds(pd.DataFrame(rows), target_fars=(0.34,))

    row = thresholds.iloc[0]
    assert row["threshold"] == 0.80
    assert row["calibration_split"] == "val"
    assert row["calibration_false_accepts"] == 1
    assert row["selected_by_val"] is True or bool(row["selected_by_val"]) is True


def test_test_metrics_do_not_participate_in_candidate_selection() -> None:
    val_metrics = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "split": "val",
                "candidate_family": "crop_overlap_probe_v1",
                "candidate_name": "research_only::crop_overlap_probe_v1:pad05",
                "target_far": 0.01,
                "tar": 0.40,
                "far": 0.0,
                "selected_by_val": True,
            },
            {
                "dataset": "toy",
                "split": "val",
                "candidate_family": "crop_overlap_probe_v1",
                "candidate_name": "research_only::crop_overlap_probe_v1:pad15",
                "target_far": 0.01,
                "tar": 0.50,
                "far": 0.0,
                "selected_by_val": True,
            },
        ]
    )
    test_metrics = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "split": "test",
                "candidate_family": "crop_overlap_probe_v1",
                "candidate_name": "research_only::crop_overlap_probe_v1:pad05",
                "target_far": 0.01,
                "tar": 1.00,
                "far": 0.0,
                "selected_by_val": True,
            }
        ]
    )

    selected = select_candidates_from_val(pd.concat([val_metrics, test_metrics], ignore_index=True), target_far=0.01)

    assert len(selected) == 1
    assert selected.iloc[0]["candidate_name"] == "research_only::crop_overlap_probe_v1:pad15"
    assert selected.iloc[0]["split"] == "val"


def test_candidate_names_are_research_only_and_script_local() -> None:
    specs = candidate_registry()

    assert_research_only_candidate_names(specs)
    assert all(spec.candidate_name.startswith("research_only::") for spec in specs)
    assert not any(spec.candidate_name in {"sift", "sift_plain_roll_v2"} for spec in specs)


def test_fusion_rules_do_not_use_test_labels() -> None:
    rows = [
        _source_row("toy", "val", 0, 1, canonical=0.1, inliers=1, v2=1.0),
        _source_row("toy", "val", 0, 2, canonical=0.2, inliers=2, v2=2.0),
        _source_row("toy", "val", 1, 3, canonical=0.9, inliers=9, v2=9.0),
        _source_row("toy", "test", 0, 4, canonical=100.0, inliers=100, v2=100.0),
        _source_row("toy", "test", 1, 5, canonical=0.3, inliers=3, v2=3.0),
    ]
    source = pd.DataFrame(rows)
    source_mutated = source.copy()
    source_mutated.loc[source_mutated["split"] == "test", "label"] = 1 - source_mutated.loc[
        source_mutated["split"] == "test", "label"
    ]

    scores = build_fusion_probe_scores(source)
    scores_mutated = build_fusion_probe_scores(source_mutated)
    thresholds = calibrate_conservative_or_thresholds(source, target_fars=(0.34,))
    thresholds_mutated = calibrate_conservative_or_thresholds(source_mutated, target_fars=(0.34,))

    pd.testing.assert_series_equal(scores["score"], scores_mutated["score"], check_names=False)
    assert thresholds.iloc[0]["source_thresholds_json"] == thresholds_mutated.iloc[0]["source_thresholds_json"]
    assert thresholds.iloc[0]["calibration_far"] == thresholds_mutated.iloc[0]["calibration_far"]


def test_crop_geometry_diagnostics_do_not_alter_existing_v2_scores() -> None:
    source = pd.DataFrame(
        [
            _source_row("toy", "val", 0, 1, canonical=0.1, inliers=1, v2=1.0),
            _source_row("toy", "test", 1, 2, canonical=0.2, inliers=2, v2=2.0),
        ]
    )
    source_before = source.copy(deep=True)
    candidate_scores = pd.DataFrame(
        [
            _candidate_row("toy", "val", 0, 1, 0.5),
            _candidate_row("toy", "test", 1, 2, 1.5),
        ]
    )
    candidate_thresholds = calibrate_scalar_candidate_thresholds(candidate_scores, target_fars=(0.5,))
    v2_thresholds = calibrate_reference_v2_thresholds(source, target_fars=(0.5,))

    _ = build_candidate_decisions(source, candidate_scores, candidate_thresholds, v2_thresholds)

    assert_v2_scores_unchanged(source_before, source)


def test_output_tables_include_sd300b_sd300c_and_frgp_5_10(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "split": "test",
                "candidate_family": "geometry_probe_v1",
                "candidate_name": "research_only::geometry_probe_v1:affine_partial_2d",
                "target_far": 0.01,
                "tar": 0.5,
                "far": 0.0,
                "selected_by_val": True,
            }
            for dataset in ("nist_sd300b", "nist_sd300c")
        ]
    )
    per_frgp = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "split": "test",
                "candidate_family": "geometry_probe_v1",
                "candidate_name": "research_only::geometry_probe_v1:affine_partial_2d",
                "target_far": 0.01,
                "frgp": frgp,
                "tar": 0.5,
                "far": 0.0,
                "selected_by_val": True,
            }
            for dataset in ("nist_sd300b", "nist_sd300c")
            for frgp in (5, 10)
        ]
    )

    assert_required_output_coverage(metrics, per_frgp)
