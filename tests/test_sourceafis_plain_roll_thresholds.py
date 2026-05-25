from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.diagnostics.run_sourceafis_plain_roll_benchmark import (
    SourceAfisBenchmarkError,
    select_threshold_for_far,
    validate_output_directory,
)


def test_threshold_selection_uses_all_val_scores_and_picks_most_permissive() -> None:
    labels = [0, 0, 0, 1]
    scores = [0.10, 0.40, 0.80, 0.50]

    selection = select_threshold_for_far(labels, scores, 0.50)

    assert selection.threshold == pytest.approx(0.50)
    assert selection.false_accepts == 1
    assert selection.actual_far == pytest.approx(1 / 3)


def test_threshold_selection_handles_ties_as_accepted_groups() -> None:
    labels = [0, 0, 0, 1]
    scores = [0.20, 0.20, 0.90, 0.95]

    near_one = select_threshold_for_far(labels, scores, 0.99)
    one = select_threshold_for_far(labels, scores, 1.0)

    assert near_one.threshold == pytest.approx(0.90)
    assert near_one.false_accepts == 1
    assert one.threshold == pytest.approx(0.20)
    assert one.false_accepts == 3


def test_threshold_selection_uses_nextafter_when_zero_far_requires_rejecting_max_tie() -> None:
    labels = [0, 0, 1]
    scores = [1.0, 1.0, 0.5]

    selection = select_threshold_for_far(labels, scores, 0.0)

    assert selection.false_accepts == 0
    assert selection.actual_far == 0.0
    assert selection.threshold > 1.0
    assert math.isclose(selection.threshold, math.nextafter(1.0, math.inf))


def test_output_directory_rejects_legacy_current_bundle(tmp_path: Path) -> None:
    repo_root = tmp_path
    forbidden = repo_root / "artifacts" / "reports" / "benchmark" / "current" / "sourceafis"

    with pytest.raises(SourceAfisBenchmarkError, match="benchmark/current"):
        validate_output_directory(forbidden, repo_root=repo_root)
