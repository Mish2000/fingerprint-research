"""Separate software tests from optional, private historical-evidence checks."""
import os
from pathlib import Path

import pytest

HISTORICAL_FILES = {
    "test_current_diagnostics_match_final_metrics.py",
    "test_current_diagnostics_not_stale.py",
    "test_current_failure_taxonomy_protocol_counts.py",
    "test_pre_final_menachem_experiments.py",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if Path(str(item.path)).name in HISTORICAL_FILES:
            item.add_marker(pytest.mark.historical_artifacts)
            if os.getenv("FPBENCH_HISTORICAL_TESTS") != "true":
                item.add_marker(pytest.mark.skip(reason=(
                    "Private historical artifacts are not bundled. Set FPBENCH_HISTORICAL_TESTS=true "
                    "to require and verify the original files; these checks do not run a benchmark."
                )))
