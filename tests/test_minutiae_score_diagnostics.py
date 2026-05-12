from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.diagnostics.analyze_minutiae_scores import analyze_scores


def test_analyze_minutiae_scores_writes_json_and_markdown(tmp_path: Path) -> None:
    scores = tmp_path / "scores_minutiae_val.csv"
    pd.DataFrame(
        [
            {
                "label": 1,
                "path_a": "a.png",
                "path_b": "b.png",
                "score": 0.72,
                "matched_minutiae": 18,
                "tentative_minutiae": 22,
                "minutiae_count_a": 42,
                "minutiae_count_b": 40,
                "endings_a": 18,
                "endings_b": 17,
                "bifurcations_a": 24,
                "bifurcations_b": 23,
                "raw_candidate_endings_a": 22,
                "raw_candidate_endings_b": 20,
                "raw_candidate_bifurcations_a": 36,
                "raw_candidate_bifurcations_b": 34,
                "saturated_by_max_minutiae_a": False,
                "saturated_by_max_minutiae_b": False,
                "extraction_quality_flags_a": "",
                "extraction_quality_flags_b": "",
            },
            {
                "label": 0,
                "path_a": "a.png",
                "path_b": "c.png",
                "score": 0.18,
                "matched_minutiae": 4,
                "tentative_minutiae": 30,
                "minutiae_count_a": 42,
                "minutiae_count_b": 96,
                "endings_a": 18,
                "endings_b": 1,
                "bifurcations_a": 24,
                "bifurcations_b": 95,
                "raw_candidate_endings_a": 22,
                "raw_candidate_endings_b": 3,
                "raw_candidate_bifurcations_a": 36,
                "raw_candidate_bifurcations_b": 260,
                "saturated_by_max_minutiae_a": False,
                "saturated_by_max_minutiae_b": True,
                "extraction_quality_flags_a": "",
                "extraction_quality_flags_b": "saturated;bifurcation_dominated",
            },
            {
                "label": 1,
                "path_a": "d.png",
                "path_b": "e.png",
                "score": 0.31,
                "matched_minutiae": 7,
                "tentative_minutiae": 12,
                "minutiae_count_a": 30,
                "minutiae_count_b": 28,
                "endings_a": 14,
                "endings_b": 13,
                "bifurcations_a": 16,
                "bifurcations_b": 15,
                "raw_candidate_endings_a": 16,
                "raw_candidate_endings_b": 14,
                "raw_candidate_bifurcations_a": 21,
                "raw_candidate_bifurcations_b": 20,
                "saturated_by_max_minutiae_a": False,
                "saturated_by_max_minutiae_b": False,
                "extraction_quality_flags_a": "",
                "extraction_quality_flags_b": "",
            },
        ]
    ).to_csv(scores, index=False)

    outdir = tmp_path / "diag"
    payload = analyze_scores(scores, outdir)

    json_path = outdir / "minutiae_score_diagnostics.json"
    md_path = outdir / "minutiae_score_diagnostics.md"
    assert json_path.exists()
    assert md_path.exists()

    disk_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert disk_payload == payload
    assert payload["n_rows"] == 3
    assert payload["label_counts"] == {"0": 1, "1": 2}
    assert len(payload["threshold_sweep"]) == 5
    assert payload["saturation_rate"]["rate"] == 1 / 6
    assert payload["quality_flag_rates"]["rates"]["saturated"]["count"] == 1
    assert payload["top_impostor_scores"][0]["score"] == 0.18
    assert payload["lowest_genuine_scores"][0]["score"] == 0.31
    assert "Threshold Sweep" in md_path.read_text(encoding="utf-8")
