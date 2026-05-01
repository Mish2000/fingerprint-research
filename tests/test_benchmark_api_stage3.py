from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import apps.api.main as api_main
from fastapi.testclient import TestClient

from apps.api.benchmark_catalog import (
    load_benchmark_runs,
    load_benchmark_summary,
    load_best_methods,
    load_comparison,
    resolve_benchmark_artifact,
)
from apps.api.main import app

client = TestClient(app)

SUMMARY_HEADER = (
    "timestamp_utc,method,split,n_pairs,auc,eer,tar_at_far_1e_2,tar_at_far_1e_3,"
    "avg_ms_pair_reported,avg_ms_pair_wall,scores_csv,meta_json,config_json"
)


def write_summary_csv(run_dir: Path, rows: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results_summary.csv").write_text(
        "\n".join([SUMMARY_HEADER, *rows]),
        encoding="utf-8",
    )


def make_summary_row(
    *,
    method: str,
    split: str,
    auc: float,
    eer: float,
    reported_ms: float | None = None,
    wall_ms: float | None = None,
    dataset: str | dict[str, str] | None = "nist_sd300b",
    pairs_path: str | None = None,
    n_pairs: int = 100,
    scores_csv: str = "",
    meta_json: str = "",
    timestamp_utc: str = "2026-04-01T00:00:00Z",
    method_semantics_epoch: str | None = None,
) -> str:
    config: dict[str, object] = {}
    if dataset is not None:
        config["dataset"] = dataset
    if pairs_path is not None:
        config["pairs_path"] = pairs_path
    if method_semantics_epoch is not None:
        config["method_semantics_epoch"] = method_semantics_epoch
    config_json = json.dumps(config).replace('"', '""')
    reported = "" if reported_ms is None else str(reported_ms)
    wall = "" if wall_ms is None else str(wall_ms)
    return (
        f'{timestamp_utc},{method},{split},{n_pairs},{auc},{eer},,,{reported},{wall},'
        f'{scores_csv},{meta_json},"{config_json}"'
    )


def write_run_manifest(run_dir: Path, dataset: str | dict[str, str]) -> None:
    payload = {"dataset": dataset}
    (run_dir / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def create_curated_benchmark_root(tmp_path: Path) -> Path:
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    run_dir = bench_root / "full_nist_sd300b_h6"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.ok").write_text("ok", encoding="utf-8")
    (run_dir / "results_summary.md").write_text("benchmark summary", encoding="utf-8")
    write_run_manifest(
        run_dir,
        {
            "name": "nist_sd300b",
            "resolved_data_dir": "C:\\fingerprint-research\\data\\processed\\nist_sd300b",
        },
    )

    for method in ("classic_v2", "sift", "dl_quick", "vit"):
        (run_dir / f"scores_{method}_test.csv").write_text("score\n0.9\n", encoding="utf-8")

    write_summary_csv(
        run_dir,
        [
            make_summary_row(
                method="classic_v2",
                split="test",
                auc=0.71,
                eer=0.29,
                wall_ms=7.2,
                pairs_path="C:\\pairs_test.csv",
            ),
            make_summary_row(
                method="sift",
                split="test",
                auc=0.91,
                eer=0.10,
                wall_ms=5.0,
                pairs_path="C:\\pairs_test.csv",
                method_semantics_epoch="sift_runtime_aligned_v1",
            ),
            make_summary_row(
                method="dl_quick",
                split="test",
                auc=0.79,
                eer=0.21,
                reported_ms=1.2,
                wall_ms=1.3,
                pairs_path="C:\\pairs_test.csv",
            ),
            make_summary_row(
                method="vit",
                split="test",
                auc=0.81,
                eer=0.16,
                reported_ms=1.5,
                wall_ms=1.6,
                pairs_path="C:\\pairs_test.csv",
            ),
        ],
    )
    return bench_root


def bind_benchmark_root(monkeypatch, bench_root: Path) -> None:
    monkeypatch.setattr(api_main, "load_benchmark_runs", partial(load_benchmark_runs, root=bench_root))
    monkeypatch.setattr(api_main, "load_benchmark_summary", partial(load_benchmark_summary, root=bench_root))
    monkeypatch.setattr(api_main, "load_comparison", partial(load_comparison, root=bench_root))
    monkeypatch.setattr(api_main, "load_best_methods", partial(load_best_methods, root=bench_root))


def test_benchmark_runs_has_curated_defaults_and_metadata(tmp_path: Path, monkeypatch):
    bench_root = create_curated_benchmark_root(tmp_path)
    bind_benchmark_root(monkeypatch, bench_root)

    response = client.get("/benchmark/runs")
    assert response.status_code == 200

    body = response.json()
    assert body["default_view_mode"] == "canonical"
    assert body["default_dataset"] == "nist_sd300b"
    assert body["default_split"] == "test"
    assert isinstance(body["runs"], list)
    assert body["runs"]

    first = body["runs"][0]
    assert first["run"] == "full_nist_sd300b_h6"
    assert "view_mode" in first
    assert "status" in first
    assert "validation_state" in first
    assert "artifact_count" in first
    assert "summary_note" in first
    assert first["methods"] == ["classic_gftt_orb", "sift", "dl", "vit"]
    assert first["benchmark_methods"] == ["classic_v2", "sift", "dl_quick", "vit"]


def test_showcase_summary_excludes_empty_or_noncanonical_entries(tmp_path: Path):
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"

    canonical_run = bench_root / "full_nist_sd300b_h6"
    canonical_run.mkdir(parents=True, exist_ok=True)
    (canonical_run / "validation.ok").write_text("ok", encoding="utf-8")
    (canonical_run / "scores_sift_test.csv").write_text("score\n0.9\n", encoding="utf-8")
    write_summary_csv(
        canonical_run,
        [
            make_summary_row(
                method="sift",
                split="test",
                auc=0.9,
                eer=0.1,
                reported_ms=4.2,
                wall_ms=5.1,
                pairs_path="C:\\pairs_test.csv",
                method_semantics_epoch="sift_runtime_aligned_v1",
            ),
        ],
    )

    incomplete_canonical_run = bench_root / "full_nist_sd300c_h6"
    incomplete_canonical_run.mkdir(parents=True, exist_ok=True)
    (incomplete_canonical_run / "validation.ok").write_text("ok", encoding="utf-8")
    write_summary_csv(
        incomplete_canonical_run,
        [
            make_summary_row(
                method="sift",
                split="val",
                auc=0.82,
                eer=0.18,
                n_pairs=120,
                pairs_path="C:\\pairs_val.csv",
                dataset="nist_sd300c",
                method_semantics_epoch="sift_runtime_aligned_v1",
            ),
        ],
    )

    smoke_run = bench_root / "smoke_nist_sd300c_h6"
    write_summary_csv(
        smoke_run,
        [
            make_summary_row(
                method="sift",
                split="val",
                auc=0.95,
                eer=0.05,
                reported_ms=1.0,
                wall_ms=1.1,
                n_pairs=20,
                pairs_path="C:\\pairs_val.csv",
                dataset="nist_sd300c",
                method_semantics_epoch="sift_runtime_aligned_v1",
            ),
        ],
    )

    summary = load_benchmark_summary(root=bench_root, view_mode="archive")

    assert summary.view_mode == "canonical"
    assert summary.dataset == "nist_sd300b"
    assert summary.split == "test"
    assert [item.key for item in summary.available_datasets] == ["nist_sd300b"]
    assert [item.key for item in summary.available_splits] == ["test"]
    assert summary.result_count == 1
    assert summary.method_count == 1
    assert summary.run_count == 1


def test_comparison_endpoint_returns_canonical_nonempty_payload_for_valid_selection(tmp_path: Path, monkeypatch):
    bench_root = create_curated_benchmark_root(tmp_path)
    bind_benchmark_root(monkeypatch, bench_root)

    response = client.get(
        "/benchmark/comparison",
        params={
            "dataset": "nist_sd300b",
            "split": "test",
            "view_mode": "archive",
            "sort_mode": "best_accuracy",
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["default_dataset"] == "nist_sd300b"
    assert body["default_split"] == "test"
    assert body["view_mode"] == "canonical"
    assert body["rows"]
    assert body["splits"]

    for row in body["rows"]:
        assert row["dataset"] == "nist_sd300b"
        assert row["split"] == "test"
        assert row["view_mode"] == "canonical"
        assert row["status"] in {"validated", "partial"}
        assert isinstance(row["artifacts"], list)
        assert "summary_text" in row
        assert row["provenance"]["run"] == row["run"]

    rows_by_benchmark_method = {row["benchmark_method"]: row for row in body["rows"]}
    assert rows_by_benchmark_method["classic_v2"]["method"] == "classic_gftt_orb"
    assert rows_by_benchmark_method["classic_v2"]["method_label"] == "Classic (ROI GFTT+ORB)"
    assert rows_by_benchmark_method["classic_v2"]["provenance"]["canonical_method"] == "classic_gftt_orb"
    assert rows_by_benchmark_method["classic_v2"]["provenance"]["benchmark_method"] == "classic_v2"
    assert rows_by_benchmark_method["classic_v2"]["provenance"]["method_label"] == "Classic (ROI GFTT+ORB)"
    assert rows_by_benchmark_method["dl_quick"]["method"] == "dl"
    assert rows_by_benchmark_method["dl_quick"]["method_label"] == "Deep Learning (ResNet50)"
    assert rows_by_benchmark_method["dl_quick"]["provenance"]["canonical_method"] == "dl"
    assert rows_by_benchmark_method["dl_quick"]["provenance"]["benchmark_method"] == "dl_quick"
    assert rows_by_benchmark_method["dl_quick"]["provenance"]["method_label"] == "Deep Learning (ResNet50)"
    assert rows_by_benchmark_method["dl_quick"]["provenance"]["benchmark_methods_in_run"] == ["classic_v2", "sift", "dl_quick", "vit"]
    assert rows_by_benchmark_method["dl_quick"]["provenance"]["methods_in_run"] == ["classic_gftt_orb", "sift", "dl", "vit"]


def test_best_method_endpoint_resolves_deterministic_winners(tmp_path: Path, monkeypatch):
    bench_root = create_curated_benchmark_root(tmp_path)
    bind_benchmark_root(monkeypatch, bench_root)

    response = client.get(
        "/benchmark/best",
        params={
            "dataset": "nist_sd300b",
            "split": "test",
            "view_mode": "archive",
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["view_mode"] == "canonical"
    entries = {entry["metric"]: entry for entry in body["entries"]}
    assert set(entries) == {"best_auc", "best_eer", "best_latency"}
    assert entries["best_auc"]["method"] == "sift"
    assert entries["best_auc"]["benchmark_method"] == "sift"
    assert entries["best_auc"]["run"] == "full_nist_sd300b_h6"
    assert entries["best_eer"]["method"] == "sift"
    assert entries["best_latency"]["run"] == "full_nist_sd300b_h6"
    assert entries["best_latency"]["method"] in {"dl", "vit"}
    assert entries["best_latency"]["benchmark_method"] in {"dl_quick", "vit"}
    if entries["best_latency"]["benchmark_method"] == "dl_quick":
        assert entries["best_latency"]["method_label"] == "Deep Learning (ResNet50)"


def test_default_showcase_row_is_deterministic(tmp_path: Path):
    bench_root = create_curated_benchmark_root(tmp_path)

    first = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="archive",
        sort_mode="lowest_latency",
        root=bench_root,
    )
    second = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="archive",
        sort_mode="lowest_latency",
        root=bench_root,
    )

    assert first.rows
    assert second.rows
    assert first.view_mode == "canonical"
    assert second.view_mode == "canonical"
    assert (first.rows[0].run, first.rows[0].method, first.rows[0].split) == (
        second.rows[0].run,
        second.rows[0].method,
        second.rows[0].split,
    )


def test_invalid_artifacts_are_ignored_without_failing(tmp_path: Path):
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    run_dir = bench_root / "full_nist_sd300b_h6"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.ok").write_text("ok", encoding="utf-8")
    (run_dir / "scores_sift_test.csv").write_text("score\n0.9\n", encoding="utf-8")
    write_summary_csv(
        run_dir,
        [
            make_summary_row(
                method="sift",
                split="test",
                auc=0.9,
                eer=0.1,
                reported_ms=4.2,
                wall_ms=5.1,
                pairs_path="C:\\pairs_test.csv",
                method_semantics_epoch="sift_runtime_aligned_v1",
            ),
        ],
    )

    tmp_run = bench_root / "tmp"
    write_summary_csv(
        tmp_run,
        [
            make_summary_row(
                method="sift",
                split="test",
                auc=0.5,
                eer=0.5,
                reported_ms=1.0,
                wall_ms=1.1,
                n_pairs=10,
                pairs_path="C:\\pairs_test.csv",
                method_semantics_epoch="sift_runtime_aligned_v1",
            ),
        ],
    )

    summary = load_benchmark_summary(root=bench_root)
    assert [item.key for item in summary.available_datasets] == ["nist_sd300b"]
    assert [item.key for item in summary.available_splits] == ["test"]

    response = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        sort_mode="best_accuracy",
        root=bench_root,
    )

    assert len(response.rows) == 1
    row = response.rows[0]
    assert row.method == "sift"
    assert row.benchmark_method == "sift"
    assert row.run == "full_nist_sd300b_h6"
    assert row.artifact_count >= 1
    assert "summary_csv" in row.available_artifacts
    assert row.provenance is not None
    assert row.provenance.pairs_path == "C:\\pairs_test.csv"
    assert all("tmp" not in item.run for item in response.rows)


def test_benchmark_artifact_lookup_still_uses_raw_benchmark_filenames(tmp_path: Path) -> None:
    bench_root = create_curated_benchmark_root(tmp_path)

    target = resolve_benchmark_artifact(
        "full_nist_sd300b_h6",
        "scores_dl_quick_test.csv",
        root=bench_root,
    )

    assert target.name == "scores_dl_quick_test.csv"
    assert target.is_file()


def test_manifest_dataset_object_is_supported_for_noncanonical_run(tmp_path: Path):
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    run_dir = bench_root / "current"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.ok").write_text("ok", encoding="utf-8")
    (run_dir / "results_summary.md").write_text("benchmark summary", encoding="utf-8")
    write_run_manifest(
        run_dir,
        {
            "name": "nist_sd300b",
            "resolved_data_dir": "C:\\fingerprint-research\\data\\processed\\nist_sd300b",
        },
    )
    (run_dir / "scores_sift_test.csv").write_text("score\n0.9\n", encoding="utf-8")
    write_summary_csv(
        run_dir,
        [
            make_summary_row(
                method="sift",
                split="test",
                auc=0.9,
                eer=0.1,
                wall_ms=4.2,
                dataset=None,
                pairs_path="C:\\pairs_test.csv",
                method_semantics_epoch="sift_runtime_aligned_v1",
            ),
        ],
    )

    response = load_benchmark_runs(root=bench_root)

    assert [item.run for item in response.runs] == ["current"]
    assert response.default_dataset == "nist_sd300b"


def test_legacy_harris_and_sift_rows_are_excluded_from_current_benchmark_surfaces(tmp_path: Path) -> None:
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    run_dir = bench_root / "full_nist_sd300b_h6"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.ok").write_text("ok", encoding="utf-8")
    (run_dir / "results_summary.md").write_text("benchmark summary", encoding="utf-8")
    write_run_manifest(
        run_dir,
        {
            "name": "nist_sd300b",
            "resolved_data_dir": "C:\\fingerprint-research\\data\\processed\\nist_sd300b",
        },
    )

    for method in ("classic_v2", "harris", "sift", "dl_quick", "vit"):
        (run_dir / f"scores_{method}_test.csv").write_text("score\n0.9\n", encoding="utf-8")

    write_summary_csv(
        run_dir,
        [
            make_summary_row(
                method="classic_v2",
                split="test",
                auc=0.71,
                eer=0.29,
                wall_ms=7.2,
                pairs_path="C:\\pairs_test.csv",
            ),
            make_summary_row(
                method="harris",
                split="test",
                auc=0.99,
                eer=0.01,
                wall_ms=4.0,
                pairs_path="C:\\pairs_test.csv",
            ),
            make_summary_row(
                method="sift",
                split="test",
                auc=0.98,
                eer=0.02,
                wall_ms=4.5,
                pairs_path="C:\\pairs_test.csv",
            ),
            make_summary_row(
                method="dl_quick",
                split="test",
                auc=0.79,
                eer=0.21,
                reported_ms=1.2,
                wall_ms=1.3,
                pairs_path="C:\\pairs_test.csv",
            ),
            make_summary_row(
                method="vit",
                split="test",
                auc=0.81,
                eer=0.16,
                reported_ms=1.5,
                wall_ms=1.6,
                pairs_path="C:\\pairs_test.csv",
            ),
        ],
    )

    runs = load_benchmark_runs(root=bench_root)
    assert len(runs.runs) == 1
    assert runs.runs[0].methods == ["classic_gftt_orb", "dl", "vit"]
    assert runs.runs[0].benchmark_methods == ["classic_v2", "dl_quick", "vit"]

    comparison = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        sort_mode="best_accuracy",
        root=bench_root,
    )
    assert {row.benchmark_method for row in comparison.rows} == {"classic_v2", "dl_quick", "vit"}

    best = load_best_methods(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        root=bench_root,
    )
    assert all(entry.benchmark_method not in {"harris", "sift"} for entry in best.entries)


def test_demo_cases_and_assets_if_available():
    response = client.get("/demo/cases")
    assert response.status_code == 200
    body = response.json()
    assert "cases" in body

    cases = body["cases"]
    if not cases:
        return

    case_id = cases[0]["id"]
    img_a = client.get(f"/demo/cases/{case_id}/a")
    img_b = client.get(f"/demo/cases/{case_id}/b")
    assert img_a.status_code == 200
    assert img_b.status_code == 200
