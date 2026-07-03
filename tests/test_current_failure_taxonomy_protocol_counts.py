from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "diagnostics"
    / "sourceafis_sift_quality_deep_fusion_v2_current_failure_taxonomy"
)
OUTCOMES_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "diagnostics"
    / "true_accept_failures_across_methods_current"
)
EXPECTED = {
    "nist_sd300b": {"pairs": 3556, "positives": 889, "negatives": 2667},
    "nist_sd300c": {"pairs": 3556, "positives": 889, "negatives": 2667},
}


def test_failure_taxonomy_pairs_use_current_test_protocol_counts() -> None:
    pairs = pd.read_csv(TAXONOMY_DIR / "failure_taxonomy_pairs.csv")
    by_dataset = pd.read_csv(TAXONOMY_DIR / "failure_taxonomy_by_dataset.csv")

    for dataset, expected in EXPECTED.items():
        sub = pairs[pairs["dataset"] == dataset]
        assert len(sub) == expected["pairs"]
        assert int((sub["label"] == 1).sum()) == expected["positives"]
        assert int((sub["label"] == 0).sum()) == expected["negatives"]

        row = by_dataset[by_dataset["dataset"] == dataset].iloc[0]
        assert int(row["pairs"]) == expected["pairs"]
        assert int(row["positives"]) == expected["positives"]
        assert int(row["negatives"]) == expected["negatives"]

    assert not (by_dataset["pairs"] == 2844).any()
    assert not (by_dataset["positives"] == 711).any()


def test_all_current_thresholds_are_calibrated_from_val_not_test() -> None:
    manifest = json.loads((TAXONOMY_DIR / "current_diagnostics_manifest.json").read_text(encoding="utf-8"))

    for threshold in manifest["thresholds"]:
        assert threshold["target_far"] == 0.01
        assert threshold["calibration_split"] == "val"
        assert threshold["calibration_negative_count"] == 2631
        assert threshold["calibration_positive_count"] == 877
        assert threshold["selection_rule"] == "lowest VAL negative-score threshold with VAL FAR <= target"

    outcomes = pd.read_csv(OUTCOMES_DIR / "all_method_outcomes.csv")
    for (dataset, method_alias), sub in outcomes.groupby(["dataset", "method_alias"]):
        assert dataset in EXPECTED
        assert len(sub) == EXPECTED[dataset]["pairs"]
        assert int((sub["label"] == 1).sum()) == EXPECTED[dataset]["positives"]
        assert int((sub["label"] == 0).sum()) == EXPECTED[dataset]["negatives"]
        assert set(sub["threshold_source_split"]) == {"val"}
