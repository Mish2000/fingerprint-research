from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.research.run_identification_self_match_experiment import (
    DEFAULT_METHODS,
    DEFAULT_RERANK_POLICY,
    ManifestExperimentRow,
    build_parser,
    ensure_safe_reset_prefix,
    load_manifest_selection,
    query_self_matches_for_method,
    resolve_experiment_methods,
    summarize_method_results,
    synthetic_national_id,
)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["dataset", "capture", "subject_id", "impression", "ppi", "frgp", "path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _experiment_row(tmp_path: Path) -> ManifestExperimentRow:
    image = tmp_path / "probe.png"
    image.write_bytes(b"probe")
    return ManifestExperimentRow(
        selected_index=1,
        manifest_row_index=1,
        dataset="toy",
        capture="plain",
        subject_id="100",
        impression="plain",
        ppi="500",
        frgp="2",
        image_path=image,
    )


class _RecordingIdentifyService:
    def __init__(self, random_id: str) -> None:
        self.random_id = random_id
        self.calls: list[dict[str, object]] = []

    def identify_from_path(self, **kwargs):
        self.calls.append(dict(kwargs))
        skipped = 1 if kwargs.get("skip_rerank") else 0
        performed = 0 if kwargs.get("skip_rerank") else 1
        candidate = SimpleNamespace(
            rank=1,
            retrieval_rank=1,
            random_id=self.random_id,
            retrieval_score=1.0,
            rerank_score=0.9 if performed else None,
            candidate_source_status="test_fixture_source",
        )
        return SimpleNamespace(
            top_candidate=candidate,
            candidates=[candidate],
            rerank_summary={"performed_count": performed, "skipped_count": skipped},
            latency_ms={"probe_embed_ms": 1.0, "shortlist_scan_ms": 2.0, "rerank_ms": 3.0, "total_ms": 6.0},
            candidate_pool_size=1,
            shortlist_size=1,
            decision=bool(performed),
            decision_status="ok",
            decision_basis="rerank" if performed else "vector_shortlist_only",
            rerank_status="rerank_performed" if performed else "skipped_vector_only_mode",
        )


def test_default_advisor_method_set_excludes_optional_and_research_methods() -> None:
    methods = resolve_experiment_methods()

    assert tuple(methods) == DEFAULT_METHODS
    assert "vit" not in methods
    assert "dedicated" not in methods
    assert "fusion_balanced_v1" not in methods


def test_parser_defaults_to_top1_rerank_policy() -> None:
    args = build_parser().parse_args([])

    assert args.rerank_policy == DEFAULT_RERANK_POLICY == "top1"


@pytest.mark.parametrize("policy", ["full", "top1", "none"])
def test_parser_accepts_rerank_policy_choices(policy: str) -> None:
    args = build_parser().parse_args(["--rerank-policy", policy])

    assert args.rerank_policy == policy


def test_method_alias_mapping_and_optional_vit() -> None:
    methods = resolve_experiment_methods("classic_v2,dl_quick,minutiae", include_vit=True)

    assert methods == ["classic_gftt_orb", "dl", "minutiae", "vit"]


def test_research_or_fusion_methods_are_rejected_for_self_match_experiment() -> None:
    with pytest.raises(ValueError, match="experimental rerank-only"):
        resolve_experiment_methods("dedicated")

    with pytest.raises(ValueError, match="Unsupported retrieval_method"):
        resolve_experiment_methods("fusion_balanced_v1")


def test_manifest_selection_skips_missing_paths_and_applies_capture_filter(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "dataset": "toy",
                "capture": "plain",
                "subject_id": "100",
                "impression": "plain",
                "ppi": "500",
                "frgp": "2",
                "path": "a.png",
            },
            {
                "dataset": "toy",
                "capture": "roll",
                "subject_id": "101",
                "impression": "roll",
                "ppi": "500",
                "frgp": "3",
                "path": "b.png",
            },
            {
                "dataset": "toy",
                "capture": "plain",
                "subject_id": "102",
                "impression": "plain",
                "ppi": "500",
                "frgp": "4",
                "path": "missing.png",
            },
        ],
    )

    selected, report = load_manifest_selection(
        dataset="toy",
        manifest_path=manifest,
        repo_root=tmp_path,
        limit=10,
        seed=123,
        capture_filter="plain",
    )

    assert [row.manifest_row_index for row in selected] == [1]
    assert selected[0].image_path == image_a
    assert selected[0].full_name == "Experiment toy 1 100 2 plain"
    assert selected[0].national_id == "000000001"
    assert selected[0].expected_random_id.startswith("selfmatch_toy_000001_")
    assert report.total_rows == 3
    assert report.capture_filtered_count == 1
    assert report.missing_path_count == 1
    assert report.valid_row_count == 1
    assert report.selected_count == 1


def test_manifest_selection_is_deterministic_for_seeded_limit(tmp_path: Path) -> None:
    rows = []
    for index in range(6):
        image = tmp_path / f"{index}.png"
        image.write_bytes(str(index).encode("ascii"))
        rows.append(
            {
                "dataset": "toy",
                "capture": "plain",
                "subject_id": str(index),
                "impression": "plain",
                "ppi": "500",
                "frgp": str(index),
                "path": image.name,
            }
        )
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    first, _ = load_manifest_selection(
        dataset="toy",
        manifest_path=manifest,
        repo_root=tmp_path,
        limit=3,
        seed=77,
    )
    second, _ = load_manifest_selection(
        dataset="toy",
        manifest_path=manifest,
        repo_root=tmp_path,
        limit=3,
        seed=77,
    )

    assert [row.manifest_row_index for row in first] == [row.manifest_row_index for row in second]
    assert len(first) == 3


@pytest.mark.parametrize(
    ("policy", "expected_skip_rerank", "expected_rerank_limit"),
    [
        ("full", False, None),
        ("top1", False, 1),
        ("none", True, 0),
    ],
)
def test_query_rerank_policy_is_passed_to_identification_service(
    tmp_path: Path,
    policy: str,
    expected_skip_rerank: bool,
    expected_rerank_limit: int | None,
) -> None:
    row = _experiment_row(tmp_path)
    service = _RecordingIdentifyService(row.expected_random_id)

    query_rows, failure_rows = query_self_matches_for_method(
        service=service,
        method="sift",
        rows=[row],
        shortlist_size=25,
        fail_fast=True,
        rerank_policy=policy,
    )

    assert failure_rows == []
    assert service.calls[0]["skip_rerank"] is expected_skip_rerank
    assert service.calls[0]["rerank_limit"] == expected_rerank_limit
    assert query_rows[0]["rerank_policy"] == policy
    assert query_rows[0]["retrieval_top1_random_id"] == row.expected_random_id
    assert query_rows[0]["final_top1_random_id"] == row.expected_random_id


def test_summary_metric_aggregation_for_self_match_rows() -> None:
    query_rows = [
        {
            "top1_is_self": True,
            "self_in_shortlist": True,
            "self_rank": 1,
            "retrieval_score_self": 1.0,
            "top1_retrieval_score": 1.0,
            "rerank_score_self": 0.91,
            "top1_rerank_score": 0.91,
            "probe_embed_ms": 10,
            "shortlist_scan_ms": 20,
            "rerank_ms": 30,
            "total_query_ms": 60,
            "rerank_performed": True,
            "candidate_source_available": True,
        },
        {
            "top1_is_self": False,
            "self_in_shortlist": True,
            "self_rank": 3,
            "retrieval_score_self": 0.75,
            "top1_retrieval_score": 0.9,
            "rerank_score_self": "",
            "top1_rerank_score": "",
            "probe_embed_ms": 12,
            "shortlist_scan_ms": 22,
            "rerank_ms": 0,
            "total_query_ms": 80,
            "rerank_performed": False,
            "candidate_source_available": False,
        },
    ]

    summary = summarize_method_results(
        dataset="toy",
        table_prefix="self_match_exp_test_",
        method="sift",
        n_selected=2,
        n_enrolled=2,
        enroll_error_count=0,
        query_rows=query_rows,
        query_error_count=1,
        score_epsilon=1e-6,
    )

    assert summary["n_queries"] == 2
    assert summary["query_error_count"] == 1
    assert summary["top1_self_match_count"] == 1
    assert summary["top1_self_match_rate"] == pytest.approx(0.5)
    assert summary["self_in_shortlist_count"] == 2
    assert summary["mean_self_rank"] == pytest.approx(2.0)
    assert summary["p95_self_rank"] == pytest.approx(3.0)
    assert summary["exact_self_vector_score_count"] == 1
    assert summary["exact_self_vector_score_rate_epsilon"] == pytest.approx(0.5)
    assert summary["mean_self_retrieval_score"] == pytest.approx(0.875)
    assert summary["min_self_retrieval_score"] == pytest.approx(0.75)
    assert summary["mean_total_query_ms"] == pytest.approx(70.0)
    assert summary["p95_total_query_ms"] == pytest.approx(80.0)
    assert summary["rerank_performed_rate"] == pytest.approx(0.5)
    assert summary["candidate_source_available_rate"] == pytest.approx(0.5)
    assert "not every successful self-query ranked itself first" in summary["notes"]


def test_summary_separates_retrieval_and_final_top1_rates() -> None:
    summary = summarize_method_results(
        dataset="toy",
        table_prefix="self_match_exp_test_",
        method="harris",
        rerank_policy="top1",
        n_selected=2,
        n_enrolled=2,
        enroll_error_count=0,
        query_rows=[
            {
                "retrieval_top1_is_self": True,
                "final_top1_is_self": True,
                "top1_is_self": True,
                "self_in_shortlist": True,
                "retrieval_rank_self": 1,
                "final_rank_self": 1,
                "reranked_candidate_count": 1,
                "skipped_rerank_candidate_count": 24,
            },
            {
                "retrieval_top1_is_self": True,
                "final_top1_is_self": False,
                "top1_is_self": False,
                "self_in_shortlist": True,
                "retrieval_rank_self": 1,
                "final_rank_self": 2,
                "reranked_candidate_count": 1,
                "skipped_rerank_candidate_count": 24,
            },
        ],
        query_error_count=0,
    )

    assert summary["rerank_policy"] == "top1"
    assert summary["retrieval_top1_self_match_count"] == 2
    assert summary["retrieval_top1_self_match_rate"] == pytest.approx(1.0)
    assert summary["final_top1_self_match_count"] == 1
    assert summary["final_top1_self_match_rate"] == pytest.approx(0.5)
    assert summary["top1_self_match_count"] == 1
    assert summary["top1_self_match_rate"] == pytest.approx(0.5)
    assert summary["mean_retrieval_rank_self"] == pytest.approx(1.0)
    assert summary["mean_final_rank_self"] == pytest.approx(1.5)
    assert summary["reranked_candidate_count_mean"] == pytest.approx(1.0)
    assert summary["skipped_rerank_candidate_count_mean"] == pytest.approx(24.0)


def test_self_match_summary_counts_rank1_success_from_fake_query_rows() -> None:
    summary = summarize_method_results(
        dataset="toy",
        table_prefix="self_match_exp_test_",
        method="dl",
        n_selected=3,
        n_enrolled=3,
        enroll_error_count=0,
        query_rows=[
            {"top1_is_self": True, "self_in_shortlist": True, "self_rank": 1},
            {"top1_is_self": True, "self_in_shortlist": True, "self_rank": 1},
            {"top1_is_self": False, "self_in_shortlist": False, "self_rank": ""},
        ],
        query_error_count=0,
    )

    assert summary["top1_self_match_count"] == 2
    assert summary["top1_self_match_rate"] == pytest.approx(2 / 3)
    assert summary["self_in_shortlist_count"] == 2
    assert summary["self_in_shortlist_rate"] == pytest.approx(2 / 3)
    assert summary["mean_self_rank"] == pytest.approx(1.0)


def test_safe_table_prefix_guard_prevents_destructive_empty_reset() -> None:
    with pytest.raises(ValueError, match="non-empty experiment table prefix"):
        ensure_safe_reset_prefix("")

    with pytest.raises(ValueError, match="suspiciously short"):
        ensure_safe_reset_prefix("x")

    assert ensure_safe_reset_prefix("self_match_exp_20260515_121314_") == "self_match_exp_20260515_121314_"


def test_synthetic_national_id_is_9_digit_and_1_based() -> None:
    assert synthetic_national_id(1) == "000000001"
    assert synthetic_national_id(5000) == "000005000"

    with pytest.raises(ValueError, match="positive"):
        synthetic_national_id(0)
