from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import scripts.diagnostics.build_sift_plain_roll_v2_failure_taxonomy as taxonomy_module
from scripts.diagnostics.build_sift_plain_roll_v2_failure_taxonomy import (
    OVERLAP_GEOMETRY_DIAGNOSTIC_COLUMNS,
    assert_frgp_focus_coverage,
    assert_no_diagnostic_decision_inputs,
    assert_taxonomy_outputs_complete,
    assert_visual_case_sheets_and_recomputed_values,
    build_aligned_test_pairs,
    build_negative_false_accept_taxonomy,
    build_overlap_geometry_diagnostics,
    build_pair_decisions,
    build_positive_failure_taxonomy,
    load_thresholds,
    parse_fingerprint_filename,
    summarize_decision_overlap,
)


def _write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(label: int, path_a: str, path_b: str, score: float, inliers: int = 4, matches: int = 12) -> dict[str, object]:
    return {
        "label": label,
        "split": "test",
        "path_a": path_a,
        "path_b": path_b,
        "score": score,
        "inliers": inliers,
        "matches": matches,
        "k1": 100,
        "k2": 100,
    }


def test_decision_overlap_counts_and_frgp_parsing_for_1000_and_2000_formats(tmp_path: Path) -> None:
    input_dir = tmp_path / "external_validation"
    dataset = "nist_sd300b"
    score_dir = input_dir / "scores" / dataset
    pairs = [
        (
            1,
            "C:/fingerprint-research/data/raw/NIST/sd300b/images/1000/png/plain/00001001_plain_1000_03.png",
            "C:/fingerprint-research/data/raw/NIST/sd300b/images/1000/png/roll/00001001_roll_1000_03.png",
        ),
        (
            1,
            "C:/fingerprint-research/data/raw/NIST/sd300c/images/2000/png/plain/00001002_plain_2000_10.png",
            "C:/fingerprint-research/data/raw/NIST/sd300c/images/2000/png/roll/00001002_roll_2000_10.png",
        ),
        (
            1,
            "C:/fingerprint-research/data/raw/NIST/sd300b/images/1000/png/plain/00001003_plain_1000_04.png",
            "C:/fingerprint-research/data/raw/NIST/sd300b/images/1000/png/roll/00001003_roll_1000_04.png",
        ),
        (
            1,
            "C:/fingerprint-research/data/raw/NIST/sd300c/images/2000/png/plain/00001004_plain_2000_05.png",
            "C:/fingerprint-research/data/raw/NIST/sd300c/images/2000/png/roll/00001004_roll_2000_05.png",
        ),
        (
            0,
            "C:/fingerprint-research/data/raw/NIST/sd300b/images/1000/png/plain/00002001_plain_1000_03.png",
            "C:/fingerprint-research/data/raw/NIST/sd300b/images/1000/png/roll/00003001_roll_1000_03.png",
        ),
        (
            0,
            "C:/fingerprint-research/data/raw/NIST/sd300c/images/2000/png/plain/00002002_plain_2000_10.png",
            "C:/fingerprint-research/data/raw/NIST/sd300c/images/2000/png/roll/00003002_roll_2000_10.png",
        ),
        (
            0,
            "C:/fingerprint-research/data/raw/NIST/sd300b/images/1000/png/plain/00002003_plain_1000_04.png",
            "C:/fingerprint-research/data/raw/NIST/sd300b/images/1000/png/roll/00003003_roll_1000_04.png",
        ),
        (
            0,
            "C:/fingerprint-research/data/raw/NIST/sd300c/images/2000/png/plain/00002004_plain_2000_05.png",
            "C:/fingerprint-research/data/raw/NIST/sd300c/images/2000/png/roll/00003004_roll_2000_05.png",
        ),
    ]
    canonical_scores = [0.4, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9, 0.1]
    v2_scores = [0.7, 0.5, 0.7, 0.2, 0.7, 0.3, 0.8, 0.2]

    _write_scores(
        score_dir / f"scores_{dataset}_sift_test.csv",
        [_row(label, path_a, path_b, score) for (label, path_a, path_b), score in zip(pairs, canonical_scores)],
    )
    _write_scores(
        score_dir / f"scores_{dataset}_sift_plain_roll_v2_test.csv",
        [_row(label, path_a, path_b, score) for (label, path_a, path_b), score in zip(pairs, v2_scores)],
    )
    pd.DataFrame(
        [
            {"dataset": dataset, "method": "sift", "variant": "current_score", "target_far": 0.01, "threshold": 0.5},
            {"dataset": dataset, "method": "sift", "variant": "inliers", "target_far": 0.01, "threshold": 5.0},
            {
                "dataset": dataset,
                "method": "sift_plain_roll_v2",
                "variant": "official_score",
                "target_far": 0.01,
                "threshold": 0.6,
            },
        ]
    ).to_csv(input_dir / "per_dataset_thresholds.csv", index=False)

    meta_1000 = parse_fingerprint_filename(pairs[0][1])
    meta_2000 = parse_fingerprint_filename(pairs[1][1])
    assert meta_1000.subject == "00001001"
    assert meta_1000.ppi == 1000
    assert meta_1000.frgp == 3
    assert meta_2000.subject == "00001002"
    assert meta_2000.ppi == 2000
    assert meta_2000.frgp == 10

    aligned = build_aligned_test_pairs(input_dir, datasets=(dataset,))
    assert set(aligned["frgp"].astype(int)) == {3, 4, 5, 10}

    decisions = build_pair_decisions(aligned, load_thresholds(input_dir), target_fars=(0.01,))
    overlap = summarize_decision_overlap(decisions).iloc[0]

    assert overlap["positive_both_accept"] == 1
    assert overlap["positive_v2_rescue"] == 1
    assert overlap["positive_v2_lost"] == 1
    assert overlap["positive_both_reject"] == 1
    assert overlap["negative_both_reject"] == 1
    assert overlap["negative_v2_new_false_accept"] == 1
    assert overlap["negative_v2_fixed_false_accept"] == 1
    assert overlap["negative_both_false_accept"] == 1

    assert_no_diagnostic_decision_inputs(decisions)
    assert not set(OVERLAP_GEOMETRY_DIAGNOSTIC_COLUMNS).intersection(decisions.columns)


def test_visual_case_and_taxonomy_assertions_cover_generated_rows(tmp_path: Path) -> None:
    outdir = tmp_path / "taxonomy"
    sheet_dir = outdir / "visual_audit_sheets"
    sheet_dir.mkdir(parents=True)
    sheet = sheet_dir / "case.png"
    cv2.imwrite(str(sheet), np.zeros((8, 8, 3), dtype=np.uint8))
    cases = pd.DataFrame(
        [
            {
                "sheet": str(sheet),
                "v2_matches": 12,
                "v2_inliers": 5,
                "v2_score": 1.5,
                "matches_recomputed": 12,
                "inliers_recomputed": 5,
                "score_recomputed": 1.5,
            }
        ]
    )
    assert_visual_case_sheets_and_recomputed_values(cases, outdir)

    decisions = pd.DataFrame(
        [
            {
                "dataset": "nist_sd300b",
                "split": "test",
                "label": 1,
                "path_a": "plain-a.png",
                "path_b": "roll-a.png",
                "target_far": 0.01,
                "v2_accepted": False,
                "v2_score_margin_ratio": -0.75,
                "v2_score": 0.1,
                "v2_matches": 4,
                "v2_inliers": 1,
                "v2_k1": 100,
                "v2_k2": 100,
            },
            {
                "dataset": "nist_sd300b",
                "split": "test",
                "label": 0,
                "path_a": "plain-b.png",
                "path_b": "roll-b.png",
                "target_far": 0.01,
                "v2_accepted": True,
                "v2_score_margin_ratio": 0.75,
                "v2_score": 9.0,
                "v2_matches": 30,
                "v2_inliers": 12,
                "v2_k1": 100,
                "v2_k2": 100,
                "sift_inliers_threshold": 12.0,
            },
        ]
    )
    build_positive_failure_taxonomy(decisions).to_csv(outdir / "v2_positive_failure_taxonomy.csv", index=False)
    build_negative_false_accept_taxonomy(decisions).to_csv(outdir / "v2_negative_false_accept_taxonomy.csv", index=False)
    assert assert_taxonomy_outputs_complete(outdir, decisions) == (1, 1)


def test_visual_case_rows_with_sheets_require_recomputed_columns(tmp_path: Path) -> None:
    outdir = tmp_path / "taxonomy"
    sheet_dir = outdir / "visual_audit_sheets"
    sheet_dir.mkdir(parents=True)
    sheet = sheet_dir / "case.png"
    cv2.imwrite(str(sheet), np.zeros((8, 8, 3), dtype=np.uint8))

    cases = pd.DataFrame(
        [
            {
                "sheet": str(sheet),
                "v2_matches": 12,
                "v2_inliers": 5,
                "v2_score": 1.5,
            }
        ]
    )

    with pytest.raises(AssertionError, match="missing recomputation columns"):
        assert_visual_case_sheets_and_recomputed_values(cases, outdir)


def test_focus_diagnostics_only_updates_run_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outdir = tmp_path / "taxonomy"
    outdir.mkdir(parents=True)
    sheet = outdir / "visual_audit_sheets" / "case.png"
    sheet.parent.mkdir(parents=True)
    cv2.imwrite(str(sheet), np.zeros((8, 8, 3), dtype=np.uint8))

    decisions = pd.DataFrame(
        [
            {
                "dataset": "nist_sd300b",
                "split": "test",
                "label": 1,
                "path_a": "plain-a.png",
                "path_b": "roll-a.png",
                "target_far": 0.01,
                "frgp": 5,
                "v2_accepted": False,
            },
            {
                "dataset": "nist_sd300b",
                "split": "test",
                "label": 0,
                "path_a": "plain-b.png",
                "path_b": "roll-b.png",
                "target_far": 0.01,
                "frgp": 10,
                "v2_accepted": True,
            },
        ]
    )
    decisions.to_csv(outdir / "aligned_test_pair_decisions.csv", index=False)
    decisions.iloc[[0]].to_csv(outdir / "v2_positive_failure_taxonomy.csv", index=False)
    decisions.iloc[[1]].to_csv(outdir / "v2_negative_false_accept_taxonomy.csv", index=False)
    pd.DataFrame(
        [
            {
                "sheet": str(sheet),
                "v2_matches": 12,
                "v2_inliers": 5,
                "v2_score": 1.5,
                "matches_recomputed": 12,
                "inliers_recomputed": 5,
                "score_recomputed": 1.5,
            }
        ]
    ).to_csv(outdir / "visual_audit_cases.csv", index=False)
    (outdir / "run_manifest.json").write_text(json.dumps({"outputs": {"existing": "kept"}}), encoding="utf-8")

    def fake_diagnostics(decisions: pd.DataFrame, target_far: float, target_size: int, blur_ksize: int) -> pd.DataFrame:
        return pd.DataFrame([{"dataset": "nist_sd300b", "frgp": 5, "label": 1}])

    def fake_focus_cases(
        decisions: pd.DataFrame,
        outdir: Path,
        args: argparse.Namespace,
        focus_frgps: tuple[int, ...],
        target_far: float,
    ) -> pd.DataFrame:
        rows = []
        for frgp in (5, 10):
            focus_sheet = outdir / f"focus_{frgp}.png"
            cv2.imwrite(str(focus_sheet), np.zeros((8, 8, 3), dtype=np.uint8))
            rows.append(
                {
                    "dataset": "nist_sd300b",
                    "focus_frgp": frgp,
                    "sheet": str(focus_sheet),
                    "v2_matches": 12,
                    "v2_inliers": 5,
                    "v2_score": 1.5,
                    "matches_recomputed": 12,
                    "inliers_recomputed": 5,
                    "score_recomputed": 1.5,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(taxonomy_module, "build_overlap_geometry_diagnostics", fake_diagnostics)
    monkeypatch.setattr(taxonomy_module, "generate_frgp_focus_visual_audit", fake_focus_cases)
    monkeypatch.setattr(taxonomy_module, "render_frgp_focus_summary", lambda *args, **kwargs: "# focus\n")
    monkeypatch.setattr(taxonomy_module, "render_overlap_geometry_summary", lambda *args, **kwargs: "# overlap\n")

    args = argparse.Namespace(
        frgp_focus_groups="5,10",
        frgp_focus_top_n=2,
        target_size=64,
        nfeatures=3000,
        blur_ksize=0,
        ratio=0.75,
        ransac_thresh=3.0,
    )
    paths = taxonomy_module.write_focus_diagnostics_from_existing(
        outdir=outdir,
        datasets=("nist_sd300b",),
        args=args,
    )

    manifest = json.loads((outdir / "run_manifest.json").read_text(encoding="utf-8"))
    focus = manifest["focus_diagnostics"]
    assert paths["manifest"] == outdir / "run_manifest.json"
    assert focus["parameters"]["frgp_focus_groups"] == [5, 10]
    assert focus["row_counts"]["frgp_focus_cases"] == 2
    assert focus["row_counts"]["overlap_geometry_diagnostics"] == 1
    assert set(focus["outputs"]) == {
        "frgp_focus_summary",
        "frgp_focus_cases",
        "overlap_geometry_diagnostics",
        "overlap_geometry_summary",
    }
    assert "commit" in focus["git"]
    assert manifest["outputs"]["existing"] == "kept"
    assert manifest["outputs"]["frgp_focus_cases"] == str(outdir / "frgp_focus_cases.csv")


def test_frgp_focus_coverage_requires_5_and_10_for_each_dataset() -> None:
    rows = []
    for dataset in ("nist_sd300b", "nist_sd300c"):
        for frgp in (5, 10):
            rows.append({"dataset": dataset, "focus_frgp": frgp, "sheet": f"{dataset}_{frgp}.png"})
    assert_frgp_focus_coverage(pd.DataFrame(rows))


def test_overlap_geometry_diagnostics_are_output_only(tmp_path: Path) -> None:
    plain = tmp_path / "plain.png"
    roll = tmp_path / "roll.png"
    img = np.full((64, 64), 255, dtype=np.uint8)
    cv2.rectangle(img, (18, 10), (45, 54), 40, thickness=-1)
    cv2.imwrite(str(plain), img)
    cv2.imwrite(str(roll), img)
    decisions = pd.DataFrame(
        [
            {
                "dataset": "nist_sd300b",
                "split": "test",
                "label": 1,
                "subject_a": "1",
                "subject_b": "1",
                "frgp": 5,
                "path_a": str(plain),
                "path_b": str(roll),
                "target_far": 0.01,
                "decision_overlap": "both_reject",
                "canonical_score": 0.0,
                "canonical_threshold": 0.5,
                "canonical_accepted": False,
                "v2_score": 0.1,
                "v2_threshold": 1.0,
                "v2_accepted": False,
                "v2_score_margin": -0.9,
                "v2_score_margin_ratio": -0.9,
                "v2_k1": 100,
                "v2_k2": 80,
                "v2_matches": 10,
                "v2_inliers": 2,
            }
        ]
    )
    assert_no_diagnostic_decision_inputs(decisions)
    diagnostics = build_overlap_geometry_diagnostics(decisions, target_size=64, blur_ksize=0)

    assert len(diagnostics) == 1
    assert bool(diagnostics.loc[0, "diagnostic_only"]) is True
    assert diagnostics.loc[0, "taxonomy"] == "low_inlier_failure"
    assert "crop_coverage_proxy" in diagnostics.columns
    assert set(OVERLAP_GEOMETRY_DIAGNOSTIC_COLUMNS).difference(diagnostics.columns) == set()
    assert not set(OVERLAP_GEOMETRY_DIAGNOSTIC_COLUMNS).intersection(decisions.columns)
