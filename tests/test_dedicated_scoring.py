from __future__ import annotations

from src.fpbench.matchers.dedicated_matcher import _geometry_aware_score


def test_geometry_changes_ranking_even_with_same_descriptor_similarity():
    weak, weak_meta = _geometry_aware_score(0.74, 0.72, 0.10, 0.05)
    strong, strong_meta = _geometry_aware_score(0.74, 0.72, 0.90, 0.85)
    assert strong > weak
    assert "raw_score" in weak_meta
    assert "raw_score" in strong_meta
