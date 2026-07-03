from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.diagnostics.analyze_true_accept_failures_across_methods import (
    MethodSpec,
    build_common_false_rejects,
    build_false_reject_sets,
    build_method_specific_false_rejects,
    build_pairwise_complementarity,
    build_positive_outcome_matrix,
    build_sanity_rerun_without_own_fr,
    parse_method_specs,
    run_analysis,
    summarize_method_outcomes,
)


def _write_method_fixture(root: Path, alias: str, benchmark_dir: str, method_id: str, rows: list[dict], threshold: float) -> MethodSpec:
    bench = root / "artifacts" / "reports" / "benchmark" / benchmark_dir
    scores_dir = bench / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(scores_dir / f"scores_ds_{method_id}_test.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": method_id,
                "dataset": "ds",
                "target_far": 0.01,
                "threshold": threshold,
                "calibration_split": "val",
            }
        ]
    ).to_csv(bench / "plain_roll_final_thresholds.csv", index=False)
    return MethodSpec(alias=alias, benchmark_dir=bench, method_id=method_id)


def _write_sourceafis_full_pairs_fixture(root: Path, rows: list[dict], threshold: float) -> None:
    bench = root / "artifacts" / "reports" / "benchmark" / "plain_roll_full_scores_v1" / "sourceafis"
    bench.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(bench / "sourceafis_plain_roll_scores_test.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "ds",
                "target_far": 0.01,
                "threshold": threshold,
                "calibration_split": "val",
            }
        ]
    ).to_csv(bench / "sourceafis_plain_roll_thresholds.csv", index=False)


def test_positive_matrix_common_and_method_specific_false_rejects(tmp_path: Path) -> None:
    base_rows = [
        {"dataset": "ds", "split": "test", "pair_id": "p1", "label": 1, "score": 0.9, "subject_a": "1", "subject_b": "1", "finger_position": "2"},
        {"dataset": "ds", "split": "test", "pair_id": "p2", "label": 1, "score": 0.4, "subject_a": "2", "subject_b": "2", "finger_position": "3"},
        {"dataset": "ds", "split": "test", "pair_id": "p3", "label": 1, "score": 0.3, "subject_a": "3", "subject_b": "3", "finger_position": "4"},
        {"dataset": "ds", "split": "test", "pair_id": "n1", "label": 0, "score": 0.2, "subject_a": "4", "subject_b": "5", "finger_position": "2"},
    ]
    other_rows = [
        {**base_rows[0], "score": 0.7},  # TA in both
        {**base_rows[1], "score": 0.8},  # base FR, other TA
        {**base_rows[2], "score": 0.2},  # FR in both
        {**base_rows[3], "score": 0.6},  # FA in other, irrelevant to positive matrix
    ]
    specs = [
        _write_method_fixture(tmp_path, "base", "bench_base", "method_base", base_rows, 0.5),
        _write_method_fixture(tmp_path, "other", "bench_other", "method_other", other_rows, 0.5),
    ]

    outputs = run_analysis(
        repo_root=tmp_path,
        outdir=tmp_path / "out",
        methods=specs,
        datasets=["ds"],
        split="test",
        target_fars=[0.01],
    )
    method_summary = outputs["method_outcome_summary.csv"]
    matrix = outputs["positive_pair_outcome_matrix.csv"]
    false_rejects = outputs["false_reject_sets_by_method.csv"]
    common = outputs["common_false_rejects_all_methods.csv"]
    method_specific = outputs["method_specific_false_rejects.csv"]
    sanity = outputs["rerun_without_own_false_rejects_metrics.csv"]
    pairwise = outputs["pairwise_complementarity_summary.csv"]

    base_summary = method_summary[method_summary["method_alias"] == "base"].iloc[0]
    other_summary = method_summary[method_summary["method_alias"] == "other"].iloc[0]
    assert int(base_summary["TA"]) == 1
    assert int(base_summary["FR"]) == 2
    assert int(other_summary["TA"]) == 2
    assert int(other_summary["FR"]) == 1

    p2 = matrix[matrix["pair_id"] == "p2"].iloc[0]
    assert p2["base_outcome"] == "FR"
    assert p2["other_outcome"] == "TA"
    p3 = matrix[matrix["pair_id"] == "p3"].iloc[0]
    assert bool(p3["all_methods_false_reject"])

    assert set(common["pair_id"]) == {"p3"}
    assert set(method_specific["pair_id"]) == {"p2"}
    assert set(false_rejects["pair_id"]) == {"p2", "p3"}
    assert sanity["TAR_after_filter"].eq(1.0).all()

    rescue = pairwise[(pairwise["base_method"] == "base") & (pairwise["other_method"] == "other")].iloc[0]
    assert int(rescue["base_FR_other_TA_rescued_by_other"]) == 1
    assert int(rescue["both_FR"]) == 1

    assert (tmp_path / "out" / "true_accept_failure_summary.md").exists()
    assert (tmp_path / "out" / "positive_pair_outcome_matrix.csv").exists()


def test_default_sourceafis_alias_uses_full_pairs_raw_scores(tmp_path: Path) -> None:
    sourceafis_rows = [
        {"dataset": "ds", "split": "test", "pair_id": "p1", "label": 1, "raw_score": 0.9, "subject_a": "1", "subject_b": "1", "finger_position": "2"},
        {"dataset": "ds", "split": "test", "pair_id": "p2", "label": 1, "raw_score": 0.4, "subject_a": "2", "subject_b": "2", "finger_position": "3"},
        {"dataset": "ds", "split": "test", "pair_id": "n1", "label": 0, "raw_score": 0.1, "subject_a": "3", "subject_b": "4", "finger_position": "4"},
    ]
    fusion_rows = [
        {**sourceafis_rows[0], "score": 0.8},
        {**sourceafis_rows[1], "score": 0.8},
        {**sourceafis_rows[2], "score": 0.2},
    ]
    _write_sourceafis_full_pairs_fixture(tmp_path, sourceafis_rows, threshold=0.5)
    sourceafis = parse_method_specs("sourceafis", repo_root=tmp_path)[0]
    fusion = _write_method_fixture(tmp_path, "fusion", "bench_fusion", "method_fusion", fusion_rows, 0.5)

    outputs = run_analysis(
        repo_root=tmp_path,
        outdir=tmp_path / "out",
        methods=[sourceafis, fusion],
        datasets=["ds"],
        split="test",
        target_fars=[0.01],
    )

    assert sourceafis.benchmark_dir == (
        tmp_path / "artifacts" / "reports" / "benchmark" / "plain_roll_full_scores_v1" / "sourceafis"
    ).resolve()
    matrix = outputs["positive_pair_outcome_matrix.csv"]
    assert set(matrix["pair_id"]) == {"p1", "p2"}
    p2 = matrix[matrix["pair_id"] == "p2"].iloc[0]
    assert p2["sourceafis_outcome"] == "FR"
    assert p2["fusion_outcome"] == "TA"
    sourceafis_summary = outputs["method_outcome_summary.csv"]
    sourceafis_summary = sourceafis_summary[sourceafis_summary["method_alias"] == "sourceafis"].iloc[0]
    assert int(sourceafis_summary["positives"]) == 2


def test_parse_custom_method_spec(tmp_path: Path) -> None:
    specs = parse_method_specs(
        "mine",
        repo_root=tmp_path,
        custom_specs=["mine=custom_dir:custom_method"],
    )
    assert len(specs) == 1
    assert specs[0].alias == "mine"
    assert specs[0].method_id == "custom_method"
    assert specs[0].benchmark_dir == (tmp_path / "artifacts" / "reports" / "benchmark" / "custom_dir").resolve()
