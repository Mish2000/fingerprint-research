from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTDIR = REPO_ROOT / "artifacts" / "reports" / "benchmark" / "pre_final_menachem_experiments_sd300_anatomical_v2"
PRIMARY_TARGET_TARS = {0.99, 0.995, 1.0}
VAL_CALIBRATED_MODE = "val_calibrated_apply_to_test"
SPLIT_ORACLE_MODE = "split_oracle_descriptive"
PUBLISHABLE_PROTOCOL_ROLE = "publishable_protocol"
DESCRIPTIVE_ORACLE_ROLE = "descriptive_oracle_only"
STATISTICAL_METHOD = "Deep Fusion v2 Statistical"
MANUAL_GROUP_WEIGHTED_METHOD = "Deep Fusion v2 Manual Group Weighted 45/15/30/10"
AUTO_GROUP_WEIGHTED_METHOD = "Deep Fusion v2 Auto Group Weighted"
COMBINED_DATASET = "combined_sd300b_sd300c"
HIGH_RECALL_METHOD_KEY_OUTPUTS = (
    "threshold_to_target_tar.csv",
    "finger_type_far_at_target_tar.csv",
    "high_recall_pair_outcomes.csv",
    "high_recall_false_reject_details.csv",
    "high_recall_false_accept_details.csv",
    "high_recall_method_comparison.csv",
)


@lru_cache(maxsize=None)
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTDIR / name)


def rounded_values(series: pd.Series) -> set[float]:
    return {round(float(value), 12) for value in series.dropna().unique()}


def comparison_row(target_tar: float) -> pd.Series:
    comparison = read_csv("high_recall_method_comparison.csv")
    rows = comparison[
        (comparison["method_label"] == STATISTICAL_METHOD)
        & (comparison["dataset"] == COMBINED_DATASET)
        & (comparison["analysis_mode"] == VAL_CALIBRATED_MODE)
        & (comparison["target_tar"].map(lambda value: abs(float(value) - target_tar) < 1e-12))
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_protocol_counts_are_stable_per_dataset_and_split() -> None:
    thresholds = read_csv("threshold_to_target_tar.csv")
    expected = {
        "val": {"pairs": 3508, "positive": 877, "negative": 2631},
        "test": {"pairs": 3556, "positive": 889, "negative": 2667},
    }

    for dataset in ("nist_sd300b", "nist_sd300c"):
        for split, counts in expected.items():
            rows = thresholds[(thresholds["dataset"] == dataset) & (thresholds["split"] == split)]
            assert set(rows["n_positive"].astype(int)) == {counts["positive"]}
            assert set(rows["n_negative"].astype(int)) == {counts["negative"]}
            assert set((rows["n_positive"] + rows["n_negative"]).astype(int)) == {counts["pairs"]}


def test_target_tar_coverage_and_finger_type_metadata() -> None:
    thresholds = read_csv("threshold_to_target_tar.csv")
    far_by_finger = read_csv("finger_type_far_at_target_tar.csv")

    assert PRIMARY_TARGET_TARS.issubset(rounded_values(thresholds["target_tar"]))
    assert PRIMARY_TARGET_TARS.issubset(rounded_values(far_by_finger["target_tar"]))

    finger_rows = far_by_finger[far_by_finger["aggregation_level"] == "finger_type"]
    assert not finger_rows["side"].isna().any()
    assert not finger_rows["finger_name"].isna().any()
    assert set(finger_rows["side"]) == {"both"}
    assert set(finger_rows["finger_name"]) == {"all"}


def test_statistical_995_combined_validation_and_100_percent_has_no_false_rejects() -> None:
    row_990 = comparison_row(0.99)
    assert int(row_990["FR"]) == 20
    assert int(row_990["FA"]) == 1377
    assert float(row_990["TAR"]) == pytest.approx(0.9887514060742407)
    assert float(row_990["FAR"]) == pytest.approx(0.2581552305961755)

    row_995 = comparison_row(0.995)
    assert int(row_995["FR"]) == 10
    assert int(row_995["FA"]) == 1675
    assert float(row_995["TAR"]) == pytest.approx(0.9943757030371203)
    assert float(row_995["FAR"]) == pytest.approx(0.3140232470941132)

    row_100 = comparison_row(1.0)
    assert int(row_100["FR"]) == 0
    assert int(row_100["FA"]) == 3858
    assert float(row_100["TAR"]) == pytest.approx(1.0)
    assert float(row_100["FAR"]) == pytest.approx(0.7232845894263217)
    false_rejects = read_csv("high_recall_false_reject_details.csv")
    statistical_100 = false_rejects[
        (false_rejects["method_label"] == STATISTICAL_METHOD)
        & false_rejects["target_tar"].map(lambda value: abs(float(value) - 1.0) < 1e-12)
    ]
    assert statistical_100.empty


def test_high_recall_outputs_do_not_use_stale_target_far_objective() -> None:
    for name in (
        "high_recall_pair_outcomes.csv",
        "high_recall_false_reject_details.csv",
        "high_recall_false_accept_details.csv",
        "high_recall_method_comparison.csv",
    ):
        columns = pd.read_csv(OUTDIR / name, nrows=5).columns
        assert "target_tar" in columns
        assert "target_far" not in columns

    high_recall_text = (OUTDIR / "high_recall_threshold_summary.md").read_text(encoding="utf-8").lower()
    final_summary_text = (OUTDIR / "pre_final_menachem_experiments_summary.md").read_text(encoding="utf-8").lower()
    assert "target_far" not in high_recall_text
    assert "finger_type_failure_details.csv" not in high_recall_text
    assert "finger_type_failure_details.csv" not in final_summary_text


def test_high_recall_outputs_have_unique_method_keys() -> None:
    for name in HIGH_RECALL_METHOD_KEY_OUTPUTS:
        frame = read_csv(name)
        label_column = "method_label" if "method_label" in frame.columns else "method"

        assert "method_key" in frame.columns, name
        assert not frame["method_key"].isna().any(), name
        assert not frame["method_key"].astype(str).str.strip().eq("").any(), name

        method_pairs = frame[[label_column, "method_key"]].drop_duplicates()
        assert (method_pairs.groupby(label_column)["method_key"].nunique() == 1).all(), name
        assert (method_pairs.groupby("method_key")[label_column].nunique() == 1).all(), name

        manual_key = set(method_pairs.loc[method_pairs[label_column] == MANUAL_GROUP_WEIGHTED_METHOD, "method_key"])
        auto_key = set(method_pairs.loc[method_pairs[label_column] == AUTO_GROUP_WEIGHTED_METHOD, "method_key"])
        assert len(manual_key) == 1, name
        assert len(auto_key) == 1, name
        assert manual_key != auto_key, name


def test_descriptive_oracle_and_publishable_protocol_are_explicitly_marked() -> None:
    thresholds = read_csv("threshold_to_target_tar.csv")
    oracle_rows = thresholds[thresholds["analysis_mode"] == SPLIT_ORACLE_MODE]
    publishable_rows = thresholds[thresholds["analysis_mode"] == VAL_CALIBRATED_MODE]

    assert set(oracle_rows["protocol_role"]) == {DESCRIPTIVE_ORACLE_ROLE}
    assert set(publishable_rows["protocol_role"]) == {PUBLISHABLE_PROTOCOL_ROLE}

    high_recall = read_csv("high_recall_method_comparison.csv")
    assert set(high_recall["analysis_mode"]) == {VAL_CALIBRATED_MODE}
    assert set(high_recall["protocol_role"]) == {PUBLISHABLE_PROTOCOL_ROLE}


def test_manifest_lists_existing_outputs_and_protocol_metadata() -> None:
    manifest = json.loads((OUTDIR / "pre_final_experiments_manifest.json").read_text(encoding="utf-8"))

    assert manifest["generated_at"]
    assert manifest["script_path"] == "scripts/diagnostics/build_pre_final_menachem_experiments.py"
    assert PRIMARY_TARGET_TARS.issubset({round(float(value), 12) for value in manifest["target_tars"]})
    assert PRIMARY_TARGET_TARS == {round(float(value), 12) for value in manifest["primary_high_recall_target_tars"]}
    assert manifest["publishable_protocol"] == VAL_CALIBRATED_MODE
    assert manifest["descriptive_only_protocol"] == SPLIT_ORACLE_MODE
    assert manifest["input_score_files"]

    for output in manifest["output_files"]:
        assert (REPO_ROOT / output).exists(), output
