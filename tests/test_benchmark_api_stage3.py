from __future__ import annotations

import csv
import json
import shutil
import sys
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
from pipelines.benchmark import validate_benchmark_bundle
from pipelines.benchmark.repair_benchmark_bundle_metadata import repair_bundle_metadata

client = TestClient(app)

SUMMARY_HEADER = (
    "timestamp_utc,method,split,n_pairs,auc,eer,tar_at_far_1e_2,tar_at_far_1e_3,"
    "avg_ms_pair_reported,avg_ms_pair_wall,scores_csv,meta_json,config_json"
)
PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfeA\xe2!\xbc\x00\x00\x00\x00IEND\xaeB`\x82"
)

H5_FULL_NIST_B = "full_nist_sd300b_h5"
H5_FULL_NIST_C = "full_nist_sd300c_h5"
H5_FULL_POLYU = "full_polyu_cross_h5"
H5_SMOKE_NIST_B = "smoke_nist_sd300b_h5"
H5_SMOKE_NIST_C = "smoke_nist_sd300c_h5"
H5_SMOKE_POLYU = "smoke_polyu_cross_h5"
H6_FULL_NIST_B = "full_nist_sd300b_h6"
H6_SMOKE_NIST_B = "smoke_nist_sd300b_h6"


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


def write_png(path: Path) -> None:
    path.write_bytes(PNG_BYTES)


def create_curated_benchmark_root(tmp_path: Path) -> Path:
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    run_dir = bench_root / H5_FULL_NIST_B
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

    for method in ("classic_v2", "minutiae", "harris", "sift", "dl_quick", "vit"):
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
                method="minutiae",
                split="test",
                auc=0.84,
                eer=0.16,
                wall_ms=8.0,
                pairs_path="C:\\pairs_test.csv",
                method_semantics_epoch="minutiae_crossing_number_aligned_v2",
            ),
            make_summary_row(
                method="harris",
                split="test",
                auc=0.84,
                eer=0.17,
                wall_ms=6.0,
                pairs_path="C:\\pairs_test.csv",
                method_semantics_epoch="harris_runtime_aligned_v1",
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


def create_live_current_and_reference_roots(tmp_path: Path) -> tuple[Path, Path]:
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    reference_root = tmp_path / "archive" / "reports" / "benchmark_reference"

    current_run = bench_root / "current"
    current_run.mkdir(parents=True, exist_ok=True)
    (current_run / "validation.ok").write_text("ok", encoding="utf-8")
    (current_run / "results_summary.md").write_text("current summary", encoding="utf-8")
    write_run_manifest(
        current_run,
        {
            "name": "nist_sd300b",
            "resolved_data_dir": "C:\\fingerprint-research\\data\\processed\\nist_sd300b",
        },
    )
    (current_run / "scores_sift_test.csv").write_text("score\n0.9\n", encoding="utf-8")
    write_summary_csv(
        current_run,
        [
            make_summary_row(
                method="sift",
                split="test",
                auc=0.88,
                eer=0.12,
                wall_ms=4.2,
                dataset=None,
                pairs_path="C:\\pairs_test.csv",
                method_semantics_epoch="sift_runtime_aligned_v1",
            ),
        ],
    )

    reference_run = reference_root / H6_FULL_NIST_B
    reference_run.mkdir(parents=True, exist_ok=True)
    (reference_run / "validation.ok").write_text("ok", encoding="utf-8")
    (reference_run / "results_summary.md").write_text("reference summary", encoding="utf-8")
    write_run_manifest(reference_run, {"name": "nist_sd300b"})
    for method in ("classic_v2", "dl_quick", "vit"):
        (reference_run / f"scores_{method}_test.csv").write_text("score\n0.9\n", encoding="utf-8")
    write_png(reference_run / "roc_dl_quick_test.png")
    write_summary_csv(
        reference_run,
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

    smoke_run = reference_root / H6_SMOKE_NIST_B
    smoke_run.mkdir(parents=True, exist_ok=True)
    (smoke_run / "validation.ok").write_text("ok", encoding="utf-8")
    (smoke_run / "results_summary.md").write_text("smoke summary", encoding="utf-8")
    write_run_manifest(smoke_run, {"name": "nist_sd300b", "limit": 200})
    for method in ("classic_v2", "dl_quick", "vit"):
        (smoke_run / f"scores_{method}_val.csv").write_text("score\n0.9\n", encoding="utf-8")
    write_summary_csv(
        smoke_run,
        [
            make_summary_row(
                method="classic_v2",
                split="val",
                auc=0.61,
                eer=0.39,
                wall_ms=7.2,
                n_pairs=20,
                pairs_path="C:\\pairs_val.csv",
            ),
            make_summary_row(
                method="dl_quick",
                split="val",
                auc=0.69,
                eer=0.31,
                reported_ms=1.2,
                wall_ms=1.3,
                n_pairs=20,
                pairs_path="C:\\pairs_val.csv",
            ),
            make_summary_row(
                method="vit",
                split="val",
                auc=0.73,
                eer=0.27,
                reported_ms=1.5,
                wall_ms=1.6,
                n_pairs=20,
                pairs_path="C:\\pairs_val.csv",
            ),
        ],
    )

    return bench_root, reference_root


def bind_benchmark_root(monkeypatch, bench_root: Path, reference_root: Path | None = None) -> None:
    monkeypatch.setattr(api_main, "load_benchmark_runs", partial(load_benchmark_runs, root=bench_root, reference_root=reference_root))
    monkeypatch.setattr(api_main, "load_benchmark_summary", partial(load_benchmark_summary, root=bench_root, reference_root=reference_root))
    monkeypatch.setattr(api_main, "load_comparison", partial(load_comparison, root=bench_root, reference_root=reference_root))
    monkeypatch.setattr(api_main, "load_best_methods", partial(load_best_methods, root=bench_root, reference_root=reference_root))
    monkeypatch.setattr(api_main, "resolve_benchmark_artifact", partial(resolve_benchmark_artifact, root=bench_root, reference_root=reference_root))


def write_minimal_valid_run(
    bench_root: Path,
    *,
    run_name: str,
    dataset: str,
    split: str,
    method: str = "sift",
) -> Path:
    run_dir = bench_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.ok").write_text("ok", encoding="utf-8")
    (run_dir / "results_summary.md").write_text("benchmark summary", encoding="utf-8")
    (run_dir / f"scores_{method}_{split}.csv").write_text("score\n0.9\n", encoding="utf-8")
    write_run_manifest(run_dir, {"name": dataset})
    epoch = None
    if method == "sift":
        epoch = "sift_runtime_aligned_v1"
    elif method == "harris":
        epoch = "harris_runtime_aligned_v1"
    write_summary_csv(
        run_dir,
        [
            make_summary_row(
                method=method,
                split=split,
                auc=0.87,
                eer=0.13,
                reported_ms=2.0,
                wall_ms=2.2,
                dataset=dataset,
                pairs_path=f"C:\\pairs_{split}.csv",
                method_semantics_epoch=epoch,
            ),
        ],
    )
    return run_dir


def test_h5_full_runs_are_accepted_as_canonical_without_dedicated(tmp_path: Path) -> None:
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    write_minimal_valid_run(bench_root, run_name=H5_FULL_NIST_B, dataset="nist_sd300b", split="test")
    write_minimal_valid_run(bench_root, run_name=H5_FULL_NIST_C, dataset="nist_sd300c", split="test")
    write_minimal_valid_run(bench_root, run_name=H5_FULL_POLYU, dataset="polyu_cross", split="test", method="dl_quick")

    runs = load_benchmark_runs(root=bench_root)
    by_name = {item.run: item for item in runs.runs}

    for run_name in (H5_FULL_NIST_B, H5_FULL_NIST_C, H5_FULL_POLYU):
        assert by_name[run_name].view_mode == "canonical"
        assert by_name[run_name].recommended is True
        assert by_name[run_name].validated is True
        assert "dedicated" not in by_name[run_name].benchmark_methods
        assert "dedicated" not in by_name[run_name].methods

    comparison = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        root=bench_root,
    )
    assert {row.run for row in comparison.rows} == {H5_FULL_NIST_B}


def test_h5_smoke_runs_are_accepted_as_smoke(tmp_path: Path) -> None:
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    write_minimal_valid_run(bench_root, run_name=H5_SMOKE_NIST_B, dataset="nist_sd300b", split="val")
    write_minimal_valid_run(bench_root, run_name=H5_SMOKE_NIST_C, dataset="nist_sd300c", split="val")
    write_minimal_valid_run(bench_root, run_name=H5_SMOKE_POLYU, dataset="polyu_cross", split="val", method="dl_quick")

    runs = load_benchmark_runs(root=bench_root)
    by_name = {item.run: item for item in runs.runs}

    for run_name in (H5_SMOKE_NIST_B, H5_SMOKE_NIST_C, H5_SMOKE_POLYU):
        assert by_name[run_name].view_mode == "smoke"
        assert by_name[run_name].status == "smoke"
        assert by_name[run_name].validated is True

    response = load_comparison(
        dataset="polyu_cross",
        split=None,
        view_mode="smoke",
        root=bench_root,
    )
    assert response.default_split == "val"
    assert {row.run for row in response.rows} == {H5_SMOKE_POLYU}


def test_reference_root_repairs_canonical_comparison_when_live_only_has_current(tmp_path: Path) -> None:
    bench_root, reference_root = create_live_current_and_reference_roots(tmp_path)

    response = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        root=bench_root,
        reference_root=reference_root,
    )

    assert response.view_mode == "canonical"
    assert response.default_dataset == "nist_sd300b"
    assert response.default_split == "test"
    assert response.rows
    assert {row.run for row in response.rows} == {H6_FULL_NIST_B}
    assert {row.provenance.benchmark_source_root for row in response.rows if row.provenance} == {"reference"}


def test_reference_root_repairs_canonical_summary_counts(tmp_path: Path) -> None:
    bench_root, reference_root = create_live_current_and_reference_roots(tmp_path)

    summary = load_benchmark_summary(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        root=bench_root,
        reference_root=reference_root,
    )

    assert summary.view_mode == "canonical"
    assert summary.result_count > 0
    assert summary.method_count > 0
    assert summary.run_count > 0
    assert [item.key for item in summary.available_datasets] == ["nist_sd300b"]
    assert [item.key for item in summary.available_splits] == ["test"]
    assert {item.key for item in summary.available_view_modes} == {"canonical", "smoke", "archive"}


def test_runs_catalog_includes_legacy_reference_and_keeps_current_archive(tmp_path: Path) -> None:
    bench_root, reference_root = create_live_current_and_reference_roots(tmp_path)

    runs = load_benchmark_runs(root=bench_root, reference_root=reference_root)
    by_name = {item.run: item for item in runs.runs}

    assert H6_FULL_NIST_B in by_name
    assert by_name[H6_FULL_NIST_B].view_mode == "canonical"
    assert by_name[H6_FULL_NIST_B].recommended is False
    assert by_name[H6_FULL_NIST_B].benchmark_source_root == "reference"
    assert by_name["current"].view_mode == "archive"
    assert by_name["current"].recommended is False
    assert by_name["current"].benchmark_source_root == "live"


def test_archive_view_mode_surfaces_current_rows(tmp_path: Path) -> None:
    bench_root, reference_root = create_live_current_and_reference_roots(tmp_path)

    response = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="archive",
        root=bench_root,
        reference_root=reference_root,
    )

    assert response.view_mode == "archive"
    assert response.rows
    assert "current" in {row.run for row in response.rows}
    assert all(row.view_mode == "archive" for row in response.rows)


def test_smoke_view_mode_surfaces_reference_smoke_rows(tmp_path: Path) -> None:
    bench_root, reference_root = create_live_current_and_reference_roots(tmp_path)

    response = load_comparison(
        dataset="nist_sd300b",
        split=None,
        view_mode="smoke",
        root=bench_root,
        reference_root=reference_root,
    )

    assert response.view_mode == "smoke"
    assert response.default_split == "val"
    assert response.rows
    assert {row.run for row in response.rows} == {H6_SMOKE_NIST_B}
    assert all(row.split == "val" for row in response.rows)


def test_reference_root_artifact_resolution_and_path_traversal_guard(tmp_path: Path) -> None:
    bench_root, reference_root = create_live_current_and_reference_roots(tmp_path)
    (reference_root / "outside.txt").write_text("do not serve", encoding="utf-8")

    target = resolve_benchmark_artifact(
        H6_FULL_NIST_B,
        "scores_dl_quick_test.csv",
        root=bench_root,
        reference_root=reference_root,
    )
    assert target == reference_root / H6_FULL_NIST_B / "scores_dl_quick_test.csv"

    try:
        resolve_benchmark_artifact(
            H6_FULL_NIST_B,
            "../outside.txt",
            root=bench_root,
            reference_root=reference_root,
        )
    except FileNotFoundError as exc:
        assert "escaped" in str(exc)
    else:
        raise AssertionError("Path traversal artifact request should have failed.")


def test_reference_root_roc_artifact_resolves_with_png_media_type(tmp_path: Path, monkeypatch) -> None:
    bench_root, reference_root = create_live_current_and_reference_roots(tmp_path)
    bind_benchmark_root(monkeypatch, bench_root, reference_root)

    target = resolve_benchmark_artifact(
        H6_FULL_NIST_B,
        "roc_dl_quick_test.png",
        root=bench_root,
        reference_root=reference_root,
    )
    assert target == reference_root / H6_FULL_NIST_B / "roc_dl_quick_test.png"

    response = client.get(f"/benchmark/artifacts/{H6_FULL_NIST_B}/roc_dl_quick_test.png")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == PNG_BYTES


def test_h5_live_canonical_wins_over_legacy_h6_reference(tmp_path: Path) -> None:
    bench_root, reference_root = create_live_current_and_reference_roots(tmp_path)
    live_run = bench_root / H5_FULL_NIST_B
    live_run.mkdir(parents=True, exist_ok=True)
    (live_run / "validation.ok").write_text("ok", encoding="utf-8")
    (live_run / "results_summary.md").write_text("live summary", encoding="utf-8")
    write_run_manifest(live_run, {"name": "nist_sd300b"})
    (live_run / "scores_vit_test.csv").write_text("score\n1.0\n", encoding="utf-8")
    write_summary_csv(
        live_run,
        [
            make_summary_row(
                method="vit",
                split="test",
                auc=0.99,
                eer=0.01,
                reported_ms=0.9,
                wall_ms=1.0,
                pairs_path="C:\\pairs_test.csv",
            ),
        ],
    )

    response = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        root=bench_root,
        reference_root=reference_root,
    )

    assert len(response.rows) == 1
    assert response.rows[0].run == H5_FULL_NIST_B
    assert response.rows[0].auc == 0.99
    assert response.rows[0].provenance is not None
    assert response.rows[0].provenance.benchmark_source_root == "live"
    assert H6_FULL_NIST_B not in {row.run for row in response.rows}


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
    assert first["run"] == H5_FULL_NIST_B
    assert "view_mode" in first
    assert "status" in first
    assert "validation_state" in first
    assert "artifact_count" in first
    assert "summary_note" in first
    assert first["methods"] == ["classic_gftt_orb", "minutiae", "harris", "sift", "dl", "vit"]
    assert first["benchmark_methods"] == ["classic_v2", "minutiae", "harris", "sift", "dl_quick", "vit"]


def test_showcase_summary_excludes_empty_or_noncanonical_entries(tmp_path: Path):
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"

    canonical_run = bench_root / H5_FULL_NIST_B
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

    incomplete_canonical_run = bench_root / H5_FULL_NIST_C
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

    smoke_run = bench_root / H5_SMOKE_NIST_C
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

    summary = load_benchmark_summary(root=bench_root, view_mode="canonical")

    assert summary.view_mode == "canonical"
    assert summary.dataset == "nist_sd300b"
    assert summary.split == "test"
    assert [item.key for item in summary.available_datasets] == ["nist_sd300b", "nist_sd300c"]
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
            "view_mode": "canonical",
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
    assert rows_by_benchmark_method["minutiae"]["method"] == "minutiae"
    assert rows_by_benchmark_method["minutiae"]["method_label"] == "Classic (Minutiae)"
    assert rows_by_benchmark_method["minutiae"]["provenance"]["canonical_method"] == "minutiae"
    assert rows_by_benchmark_method["minutiae"]["provenance"]["benchmark_method"] == "minutiae"
    assert rows_by_benchmark_method["dl_quick"]["method"] == "dl"
    assert rows_by_benchmark_method["dl_quick"]["method_label"] == "Deep Learning (ResNet18)"
    assert rows_by_benchmark_method["dl_quick"]["provenance"]["canonical_method"] == "dl"
    assert rows_by_benchmark_method["dl_quick"]["provenance"]["benchmark_method"] == "dl_quick"
    assert rows_by_benchmark_method["dl_quick"]["provenance"]["method_label"] == "Deep Learning (ResNet18)"
    assert rows_by_benchmark_method["dl_quick"]["provenance"]["benchmark_methods_in_run"] == ["classic_v2", "minutiae", "harris", "sift", "dl_quick", "vit"]
    assert rows_by_benchmark_method["dl_quick"]["provenance"]["methods_in_run"] == ["classic_gftt_orb", "minutiae", "harris", "sift", "dl", "vit"]


def test_best_method_endpoint_resolves_deterministic_winners(tmp_path: Path, monkeypatch):
    bench_root = create_curated_benchmark_root(tmp_path)
    bind_benchmark_root(monkeypatch, bench_root)

    response = client.get(
        "/benchmark/best",
        params={
            "dataset": "nist_sd300b",
            "split": "test",
            "view_mode": "canonical",
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["view_mode"] == "canonical"
    entries = {entry["metric"]: entry for entry in body["entries"]}
    assert set(entries) == {"best_auc", "best_eer", "best_latency"}
    assert entries["best_auc"]["method"] == "sift"
    assert entries["best_auc"]["benchmark_method"] == "sift"
    assert entries["best_auc"]["run"] == H5_FULL_NIST_B
    assert entries["best_eer"]["method"] == "sift"
    assert entries["best_latency"]["run"] == H5_FULL_NIST_B
    assert entries["best_latency"]["method"] in {"dl", "vit"}
    assert entries["best_latency"]["benchmark_method"] in {"dl_quick", "vit"}
    if entries["best_latency"]["benchmark_method"] == "dl_quick":
        assert entries["best_latency"]["method_label"] == "Deep Learning (ResNet18)"


def test_default_showcase_row_is_deterministic(tmp_path: Path):
    bench_root = create_curated_benchmark_root(tmp_path)

    first = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        sort_mode="lowest_latency",
        root=bench_root,
    )
    second = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
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
    run_dir = bench_root / H5_FULL_NIST_B
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
    assert row.run == H5_FULL_NIST_B
    assert row.artifact_count >= 1
    assert "summary_csv" in row.available_artifacts
    assert row.provenance is not None
    assert row.provenance.pairs_path == "C:\\pairs_test.csv"
    assert all("tmp" not in item.run for item in response.rows)


def test_benchmark_artifact_lookup_still_uses_raw_benchmark_filenames(tmp_path: Path) -> None:
    bench_root = create_curated_benchmark_root(tmp_path)

    target = resolve_benchmark_artifact(
        H5_FULL_NIST_B,
        "scores_dl_quick_test.csv",
        root=bench_root,
    )

    assert target.name == "scores_dl_quick_test.csv"
    assert target.is_file()


def test_dedicated_archive_rows_carry_showcase_exclusion_note(tmp_path: Path) -> None:
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    run_dir = bench_root / "full_nist_sd300b_dedicated_audit"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results_summary.md").write_text("dedicated audit", encoding="utf-8")
    (run_dir / "scores_dedicated_val.csv").write_text("score\n0.5\n", encoding="utf-8")
    write_run_manifest(run_dir, {"name": "nist_sd300b"})
    write_summary_csv(
        run_dir,
        [
            make_summary_row(
                method="dedicated",
                split="val",
                auc=0.4676472222,
                eer=0.5075,
                reported_ms=243.0,
                wall_ms=255.0,
                dataset="nist_sd300b",
                pairs_path="C:\\pairs_val.csv",
            ),
        ],
    )

    response = load_comparison(
        dataset="nist_sd300b",
        split="val",
        view_mode="archive",
        root=bench_root,
    )

    assert response.rows
    row = response.rows[0]
    assert row.benchmark_method == "dedicated"
    assert row.method_status == "experimental"
    assert row.presentation_tier == "research"
    assert row.showcase_eligible is False
    assert row.not_champion_candidate is True
    assert row.showcase_exclusion_note
    assert row.provenance is not None
    assert row.provenance.showcase_eligible is False
    assert row.provenance.research_track is True
    assert "experimental research method" in (row.provenance.showcase_exclusion_note or "")
    assert "must not be promoted" in (row.provenance.showcase_exclusion_note or "")


def test_dedicated_cannot_be_selected_as_canonical_champion(tmp_path: Path) -> None:
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    run_dir = bench_root / H5_FULL_NIST_B
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.ok").write_text("ok", encoding="utf-8")
    (run_dir / "results_summary.md").write_text("benchmark summary", encoding="utf-8")
    write_run_manifest(run_dir, {"name": "nist_sd300b"})
    for method in ("sift", "dedicated"):
        (run_dir / f"scores_{method}_test.csv").write_text("score\n0.9\n", encoding="utf-8")

    write_summary_csv(
        run_dir,
        [
            make_summary_row(
                method="dedicated",
                split="test",
                auc=0.99,
                eer=0.01,
                wall_ms=1.0,
                pairs_path="C:\\pairs_test.csv",
            ),
            make_summary_row(
                method="sift",
                split="test",
                auc=0.80,
                eer=0.20,
                wall_ms=4.0,
                pairs_path="C:\\pairs_test.csv",
                method_semantics_epoch="sift_runtime_aligned_v1",
            ),
        ],
    )

    comparison = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        sort_mode="best_accuracy",
        root=bench_root,
    )
    rows_by_method = {row.benchmark_method: row for row in comparison.rows}

    assert set(rows_by_method) == {"dedicated", "sift"}
    assert rows_by_method["dedicated"].showcase_eligible is False
    assert rows_by_method["dedicated"].auc_rank is None
    assert rows_by_method["sift"].auc_rank == 1

    best = load_best_methods(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        root=bench_root,
    )

    assert best.entries
    assert all(entry.benchmark_method != "dedicated" for entry in best.entries)


def test_missing_roc_artifact_is_not_marked_available(tmp_path: Path) -> None:
    bench_root = create_curated_benchmark_root(tmp_path)

    response = load_comparison(
        dataset="nist_sd300b",
        split="test",
        view_mode="canonical",
        sort_mode="best_accuracy",
        root=bench_root,
    )

    assert response.rows
    for row in response.rows:
        roc_artifact = next(item for item in row.artifacts if item.key == "roc_png")
        assert roc_artifact.available is False
        assert roc_artifact.url is None
        assert "roc_png" not in row.available_artifacts


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
    run_dir = bench_root / H5_FULL_NIST_B
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


def test_repair_benchmark_bundle_metadata_syncs_promoted_config_json(tmp_path: Path, monkeypatch) -> None:
    bench_root = tmp_path / "artifacts" / "reports" / "benchmark"
    staging = bench_root / "_regen_h5_tmp_20260506_182726" / "smoke_unit_h5"
    final = bench_root / "smoke_unit_h5"
    method = "dl_quick"
    split = "val"

    staging.mkdir(parents=True)
    scores_csv = staging / f"scores_{method}_{split}.csv"
    scores_csv.write_text("label,score\n1,0.9\n0,0.1\n", encoding="utf-8")
    (staging / f"scores_{method}_{split}.meta.json").write_text('{"source": "unit"}\n', encoding="utf-8")
    write_png(staging / f"roc_{method}_{split}.png")

    config = {
        "schema_version": "v2_benchmark_eval_config",
        "method": method,
        "split": split,
        "limit": 0,
        "dataset": "unit",
        "resolved_data_dir": str(tmp_path / "data"),
        "manifest_path": str(tmp_path / "data" / "manifest.csv"),
        "pairs_path": str(tmp_path / "data" / "pairs_val.csv"),
        "fusion": {
            "source_dir": str(staging),
            "fit_split": "val",
            "weights": {"sift": 0.91, "dl_quick": 0.05, "vit": 0.04},
        },
    }
    row = {
        "timestamp_utc": "2026-05-06T00:00:00Z",
        "method": method,
        "split": split,
        "n_pairs": "2",
        "auc": "1.0",
        "eer": "0.0",
        "tar_at_far_1e_2": "1.0",
        "tar_at_far_1e_3": "1.0",
        "avg_ms_pair_reported": "1.2",
        "avg_ms_pair_wall": "1.3",
        "scores_csv": str(scores_csv),
        "meta_json": str(staging / f"scores_{method}_{split}.meta.json"),
        "config_json": json.dumps(config, ensure_ascii=False),
    }
    with (staging / "results_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_HEADER.split(","))
        writer.writeheader()
        writer.writerow(row)

    run_meta = {
        "schema_version": "v2_benchmark_run_meta",
        "row": dict(row),
        "scores_csv": str(scores_csv),
        "roc_png": str(staging / f"roc_{method}_{split}.png"),
        "summary_csv": str(staging / "results_summary.csv"),
        "method_meta_json": row["meta_json"],
        "resolved_data_dir": config["resolved_data_dir"],
        "manifest_path": config["manifest_path"],
        "pairs_path": config["pairs_path"],
        "config": config,
    }
    (staging / f"run_{method}_{split}.meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (staging / "run_manifest.json").write_text(
        json.dumps({"outdir": str(staging), "fusion": {"source_dir": str(staging)}}, indent=2),
        encoding="utf-8",
    )
    shutil.copytree(staging, final)

    report = repair_bundle_metadata(final)

    repaired_rows = list(csv.DictReader((final / "results_summary.csv").open(encoding="utf-8")))
    repaired_row = repaired_rows[0]
    repaired_config = json.loads(repaired_row["config_json"])
    repaired_meta = json.loads((final / f"run_{method}_{split}.meta.json").read_text(encoding="utf-8"))

    assert report.changed_files
    assert repaired_row["scores_csv"] == str(final / f"scores_{method}_{split}.csv")
    assert repaired_row["meta_json"] == str(final / f"scores_{method}_{split}.meta.json")
    assert repaired_config["fusion"]["source_dir"] == str(final)
    assert repaired_meta["row"]["config_json"] == repaired_row["config_json"]
    assert repaired_meta["config"] == repaired_config

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_benchmark_bundle",
            "--outdir",
            str(final),
            "--expected_methods",
            method,
            "--expected_splits",
            split,
        ],
    )
    assert validate_benchmark_bundle.main() == 0

    for path in final.rglob("*"):
        if path.is_file() and ".bak" not in path.name and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            assert "_regen_h5_tmp_" not in path.read_text(encoding="utf-8")


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
