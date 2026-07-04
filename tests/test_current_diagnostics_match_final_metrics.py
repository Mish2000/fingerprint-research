from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_BENCHMARK = (
    PROJECT_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "sourceafis_sift_quality_deep_fusion_v2_statistical_anatomical_v2_ddpdeep"
)
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
EXPECTED_FUSION = {
    "nist_sd300b": {"TA": 753, "FR": 136, "FA": 23, "TR": 2644, "TAR": 753 / 889, "FAR": 23 / 2667},
    "nist_sd300c": {"TA": 757, "FR": 132, "FA": 24, "TR": 2643, "TAR": 757 / 889, "FAR": 24 / 2667},
}


def test_current_fusion_v2_outcomes_match_expected_final_metrics() -> None:
    summary = pd.read_csv(OUTCOMES_DIR / "method_outcome_summary.csv")
    taxonomy = pd.read_csv(TAXONOMY_DIR / "failure_taxonomy_by_dataset.csv")

    for dataset, expected in EXPECTED_FUSION.items():
        row = summary[(summary["method_alias"] == "fusion_v2") & (summary["dataset"] == dataset)].iloc[0]
        assert int(row["positives"]) == 889
        assert int(row["negatives"]) == 2667
        for column in ("TA", "FR", "FA", "TR"):
            assert int(row[column]) == expected[column]
        assert float(row["TAR"]) == pytest.approx(expected["TAR"], abs=1e-12)
        assert float(row["FAR"]) == pytest.approx(expected["FAR"], abs=1e-12)

        tax_row = taxonomy[taxonomy["dataset"] == dataset].iloc[0]
        assert int(tax_row["fusion_TA"]) == expected["TA"]
        assert int(tax_row["fusion_FR"]) == expected["FR"]
        assert int(tax_row["fusion_FA"]) == expected["FA"]
        assert int(tax_row["fusion_TR"]) == expected["TR"]
        assert float(tax_row["fusion_TAR"]) == pytest.approx(expected["TAR"], abs=1e-12)
        assert float(tax_row["fusion_FAR"]) == pytest.approx(expected["FAR"], abs=1e-12)


def test_current_fusion_v2_outcomes_match_canonical_final_metrics_file() -> None:
    current = pd.read_csv(OUTCOMES_DIR / "method_outcome_summary.csv")
    final_metrics = pd.read_csv(CANONICAL_BENCHMARK / "plain_roll_final_metrics.csv")

    for dataset in EXPECTED_FUSION:
        current_row = current[(current["method_alias"] == "fusion_v2") & (current["dataset"] == dataset)].iloc[0]
        final_row = final_metrics[
            (final_metrics["method"] == "sourceafis_sift_quality_deep_fusion_v2")
            & (final_metrics["dataset"] == dataset)
            & (final_metrics["split"] == "test")
            & (final_metrics["target_far"] == 0.01)
        ].iloc[0]

        assert float(current_row["threshold"]) == pytest.approx(float(final_row["threshold"]), abs=1e-12)
        assert int(current_row["TA"]) == int(final_row["TA"])
        assert int(current_row["FR"]) == int(final_row["FR"])
        assert int(current_row["FA"]) == int(final_row["FA"])
        assert int(current_row["TR"]) == int(final_row["TR"])
        assert float(current_row["TAR"]) == pytest.approx(float(final_row["TAR"]), abs=1e-12)
        assert float(current_row["FAR"]) == pytest.approx(float(final_row["FAR"]), abs=1e-12)
