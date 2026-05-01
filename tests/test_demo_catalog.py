from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from scipy import io as scipy_io

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fpbench.catalog.demo_catalog as demo_catalog
from fpbench.catalog.demo_catalog import (
    CATALOG_VERSION,
    DatasetBundle,
    MATERIALIZED_ASSET_KIND,
    build_catalog_bundle,
    build_identification_scenarios,
    build_identity_record,
    find_asset,
    IDENTITY_DATASET_ORDER,
    materialize_asset_dict,
    select_case_row,
)



def _collect_asset_dicts(catalog: dict) -> list[dict]:
    assets: list[dict] = []
    for case in catalog["verify_cases"]:
        assets.extend([case["image_a"], case["image_b"]])
    for identity in catalog["identify_gallery"]["identities"]:
        assets.extend(identity["exemplars"])
    for scenario in catalog["identify_gallery"]["demo_scenarios"]:
        assets.append(scenario["probe_asset"])
    for entry in catalog["dataset_browser_seed"]:
        assets.extend(entry["items"])
    return assets


def _require_catalog_source_data() -> None:
    missing = [
        str(ROOT / "data" / "manifests" / dataset / "manifest.csv")
        for dataset in IDENTITY_DATASET_ORDER
        if not (ROOT / "data" / "manifests" / dataset / "manifest.csv").exists()
    ]
    if missing:
        pytest.skip(f"catalog integration data is unavailable in this workspace snapshot: {missing[0]}")


def _configure_temp_catalog_roots(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.setattr(demo_catalog, "ROOT", repo_root)
    monkeypatch.setattr(demo_catalog, "MANIFESTS_ROOT", repo_root / "data" / "manifests")
    monkeypatch.setattr(demo_catalog, "BENCH_ROOT", repo_root / "artifacts" / "reports" / "benchmark")
    monkeypatch.setattr(demo_catalog, "SAMPLES_ROOT", repo_root / "data" / "samples")
    monkeypatch.setattr(demo_catalog, "ASSETS_ROOT", repo_root / "data" / "samples" / "assets")
    monkeypatch.setattr(
        demo_catalog,
        "BEST_METHODS_JSON",
        repo_root / "artifacts" / "reports" / "benchmark" / "april_comparison" / "best_methods.json",
    )


def _stub_catalog_postprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo_catalog, "materialize_catalog_assets", lambda payload: payload)
    monkeypatch.setattr(
        demo_catalog,
        "validate_catalog_payload",
        lambda payload, schema: {
            "validation_status": "pass",
            "schema_errors_count": 0,
            "validation_errors_count": 0,
            "validation_warnings_count": 0,
            "errors": [],
            "warnings": [],
            "checks": {},
            "selection_diagnostics": {
                "catalog_build_health": {},
                "case_selection": [],
                "dataset_verify_selection": [],
            },
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_minimal_nist_catalog_fixture(repo_root: Path, dataset: str = "nist_sd300b") -> dict[str, str]:
    manifests_root = repo_root / "data" / "manifests" / dataset
    manifests_root.mkdir(parents=True, exist_ok=True)
    raw_root = repo_root / "data" / "raw" / dataset
    raw_root.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "plain_subject_1": raw_root / "subject1_plain.png",
        "roll_subject_1": raw_root / "subject1_roll.png",
        "plain_subject_2": raw_root / "subject2_plain.png",
        "roll_subject_2": raw_root / "subject2_roll.png",
    }
    for index, path in enumerate(source_paths.values(), start=1):
        image = np.full((24, 24), 32 * index, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)

    manifest = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "capture": "plain",
                "subject_id": 1,
                "impression": "capture_01",
                "ppi": 1000,
                "frgp": 1,
                "path": str(source_paths["plain_subject_1"]),
                "split": "val",
                "sample_id": 1,
                "session": 1,
                "source_modality": "contact_based",
            },
            {
                "dataset": dataset,
                "capture": "roll",
                "subject_id": 1,
                "impression": "capture_01",
                "ppi": 1000,
                "frgp": 1,
                "path": str(source_paths["roll_subject_1"]),
                "split": "val",
                "sample_id": 2,
                "session": 1,
                "source_modality": "contact_based",
            },
            {
                "dataset": dataset,
                "capture": "plain",
                "subject_id": 2,
                "impression": "capture_01",
                "ppi": 1000,
                "frgp": 1,
                "path": str(source_paths["plain_subject_2"]),
                "split": "val",
                "sample_id": 3,
                "session": 1,
                "source_modality": "contact_based",
            },
            {
                "dataset": dataset,
                "capture": "roll",
                "subject_id": 2,
                "impression": "capture_01",
                "ppi": 1000,
                "frgp": 1,
                "path": str(source_paths["roll_subject_2"]),
                "split": "val",
                "sample_id": 4,
                "session": 1,
                "source_modality": "contact_based",
            },
        ]
    )
    manifest.to_csv(manifests_root / "manifest.csv", index=False)

    pairs = pd.DataFrame(
        [
            {
                "pair_id": 1,
                "label": 1,
                "split": "val",
                "subject_a": 1,
                "subject_b": 1,
                "frgp": 1,
                "path_a": manifest.iloc[0]["path"],
                "path_b": manifest.iloc[1]["path"],
            },
            {
                "pair_id": 2,
                "label": 0,
                "split": "val",
                "subject_a": 1,
                "subject_b": 2,
                "frgp": 1,
                "path_a": manifest.iloc[0]["path"],
                "path_b": manifest.iloc[3]["path"],
            },
        ]
    )
    pairs.to_csv(manifests_root / "pairs_val.csv", index=False)
    _write_json(manifests_root / "stats.json", {"manifest_rows": len(manifest), "unique_subjects": 2})
    _write_json(manifests_root / "split.json", {"splits": ["val"]})

    return {
        "plain_subject_1": str(manifest.iloc[0]["path"]),
        "roll_subject_1": str(manifest.iloc[1]["path"]),
        "roll_subject_2": str(manifest.iloc[3]["path"]),
    }


def _patch_minimal_catalog_config(monkeypatch: pytest.MonkeyPatch, dataset: str = "nist_sd300b") -> None:
    monkeypatch.setattr(demo_catalog, "IDENTITY_DATASET_ORDER", [dataset])
    monkeypatch.setattr(
        demo_catalog,
        "VERIFY_CASE_PLANS",
        [
            {
                "case_id": "fixture_verify_case",
                "dataset": dataset,
                "split": "val",
                "case_type": "easy_genuine",
                "label": 1,
                "difficulty": "easy",
                "selector": "benchmark_top",
                "demo_safe": True,
            }
        ],
    )
    monkeypatch.setattr(demo_catalog, "VERIFY_DEMO_DATASETS", frozenset({dataset}))



def test_demo_catalog_builds_non_empty_sections():
    _require_catalog_source_data()
    bundle = build_catalog_bundle(write_files=True, generated_at="2026-03-31T00:00:00Z")
    catalog = bundle["catalog"]
    assert catalog["catalog_version"] == CATALOG_VERSION
    assert len(catalog["verify_cases"]) >= 7
    assert len(catalog["identify_gallery"]["identities"]) >= 6
    assert len(catalog["dataset_browser_seed"]) >= 6
    assert {identity["dataset"] for identity in catalog["identify_gallery"]["identities"]} >= {
        "nist_sd300b",
        "nist_sd300c",
        "polyu_cross",
        "unsw_2d3d",
        "polyu_3d",
        "l3_sf_v2",
    }



def test_demo_catalog_schema_and_ids_are_valid():
    _require_catalog_source_data()
    bundle = build_catalog_bundle(write_files=True, generated_at="2026-03-31T00:00:00Z")
    catalog = bundle["catalog"]
    schema = bundle["schema"]
    Draft202012Validator(schema).validate(catalog)

    case_ids = [item["case_id"] for item in catalog["verify_cases"]]
    identity_ids = [item["identity_id"] for item in catalog["identify_gallery"]["identities"]]
    assert len(case_ids) == len(set(case_ids))
    assert len(identity_ids) == len(set(identity_ids))



def test_demo_catalog_has_required_demo_coverage():
    _require_catalog_source_data()
    bundle = build_catalog_bundle(write_files=True, generated_at="2026-03-31T00:00:00Z")
    catalog = bundle["catalog"]
    case_ids = {item["case_id"] for item in catalog["verify_cases"]}
    assert case_ids == {
        "easy_genuine_nist_1000",
        "hard_genuine_nist_1000",
        "hard_impostor_nist_1000",
        "easy_genuine_nist_2000",
        "hard_impostor_nist_2000",
        "cross_modality_genuine_crossfingerprint",
        "cross_modality_impostor_crossfingerprint",
    }

    scenario_types = {item["scenario_type"] for item in catalog["identify_gallery"]["demo_scenarios"]}
    assert scenario_types == {"positive_identification", "difficult_identification", "no_match"}

    source_notes = {entry["dataset"]: entry["notes"] for entry in catalog["source_datasets"]}
    for dataset in {"unsw_2d3d", "polyu_3d", "l3_sf_v2"}:
        assert any("not exposed as one of the 7 curated verify demo cases" in note for note in source_notes[dataset])



def test_every_curated_asset_exists_on_disk_and_report_is_clean():
    _require_catalog_source_data()
    bundle = build_catalog_bundle(write_files=True, generated_at="2026-03-31T00:00:00Z")
    catalog = bundle["catalog"]
    report = bundle["report"]

    for asset in _collect_asset_dicts(catalog):
        rel_path = asset.get("relative_path") or asset["path"]
        assert rel_path, asset["asset_id"]
        assert (ROOT / rel_path).exists(), rel_path
        assert asset["availability_status"] == "available"

    assert report["validation_status"] == "pass"
    assert report["validation_errors_count"] == 0
    if catalog["metadata"]["catalog_build_health"]["status"] == "healthy":
        assert report["validation_warnings_count"] == 0
    else:
        assert report["validation_warnings_count"] == len(report["warnings"])
        assert report["validation_warnings_count"] > 0
        assert report["selection_diagnostics"]["catalog_build_health"]["status"] == "degraded"



def test_regression_all_curated_assets_are_marked_available_and_catalog_files_are_written():
    _require_catalog_source_data()
    bundle = build_catalog_bundle(write_files=True, generated_at="2026-03-31T00:00:00Z")
    catalog = bundle["catalog"]
    assert catalog["metadata"]["validation_status"] == "pass"
    assert catalog["metadata"]["validation_warnings_count"] == len(bundle["report"]["warnings"])

    for asset in _collect_asset_dicts(catalog):
        assert asset["availability_status"] == "available"
        assert asset["source_path"], asset["asset_id"]
        assert asset["traceability"]["materialized_asset_path"] == asset["path"]

    report_path = ROOT / "data/samples/catalog.validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["validation_status"] == "pass"
    assert report["validation_warnings_count"] == len(report["warnings"])
    assert report["selection_diagnostics"]["catalog_build_health"] == catalog["metadata"]["catalog_build_health"]


def test_regression_nist_plain_roll_cases_are_same_modality_and_only_cross_datasets_emit_cross_modality():
    _require_catalog_source_data()
    bundle = build_catalog_bundle(write_files=True, generated_at="2026-03-31T00:00:00Z")
    catalog = bundle["catalog"]

    nist_cases = [case for case in catalog["verify_cases"] if case["dataset"] in {"nist_sd300b", "nist_sd300c"}]
    assert nist_cases, "expected curated NIST verify cases"
    assert all(case["modality_relation"] == "same_modality" for case in nist_cases)

    cross_cases = [case for case in catalog["verify_cases"] if case["modality_relation"] == "cross_modality"]
    assert cross_cases, "expected at least one true cross-modality verify case"
    assert {case["dataset"] for case in cross_cases} <= {"polyu_cross", "unsw_2d3d"}


def test_regression_benchmark_selection_handles_alias_raw_roots_for_polyu_cross(
    tmp_path: Path,
    monkeypatch,
):
    bench_root = tmp_path / "benchmark"
    run_root = bench_root / "full_polyu_cross_h5"
    run_root.mkdir(parents=True, exist_ok=True)

    pairs = pd.DataFrame(
        [
            {
                "pair_id": 1,
                "label": 1,
                "split": "val",
                "subject_a": 24,
                "subject_b": 24,
                "frgp": 0,
                "path_a": "C:/fingerprint-research/data/raw/PolyU_Hong_Kong/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images/first_session/p24/p1.bmp",
                "path_b": "C:/fingerprint-research/data/raw/PolyU_Hong_Kong/Cross_Fingerprint_Images_Database/contact-based_fingerprints/first_session/24_1.jpg",
            }
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "label": 1,
                "split": "val",
                "path_a": "C:/fingerprint-research/data/raw/polyu_cross/processed_contactless_2d_fingerprint_images/first_session/p24/p1.bmp",
                "path_b": "C:/fingerprint-research/data/raw/polyu_cross/contact-based_fingerprints/first_session/24_1.jpg",
                "score": 0.91,
            }
        ]
    )
    scores.to_csv(run_root / "scores_dl_quick_val.csv", index=False)

    monkeypatch.setattr("fpbench.catalog.demo_catalog.BENCH_ROOT", bench_root)
    bundle = DatasetBundle(
        dataset="polyu_cross",
        manifest=pd.DataFrame(),
        pairs_by_split={"val": pairs},
        stats={},
        split_meta={},
        protocol_note_path=None,
        manifest_lookup={},
        manifest_lookup_by_signature={},
        benchmark_best={"polyu_cross:val": {"best_auc": {"method": "dl_quick", "run": "full_polyu_cross_h5"}}},
    )

    selection = select_case_row(
        bundle,
        {
            "dataset": "polyu_cross",
            "split": "val",
            "case_type": "cross_modality_genuine",
            "label": 1,
            "difficulty": "medium",
            "selector": "benchmark_top",
        },
    )

    assert selection is not None
    assert selection.score == 0.91
    assert selection.benchmark_context["benchmark_pair_match_signature_depth"] == 3
    assert selection.selection_diagnostics["chosen_raw_benchmark_method"] == "dl_quick"
    assert selection.selection_diagnostics["chosen_canonical_method"] == "dl"
    assert selection.selection_diagnostics["benchmark_selection_status"] == "benchmark_score_used"


def test_regression_benchmark_selection_falls_back_to_heuristics_when_no_pair_overlap(
    tmp_path: Path,
    monkeypatch,
):
    bench_root = tmp_path / "benchmark"
    run_root = bench_root / "full_polyu_cross_h5"
    run_root.mkdir(parents=True, exist_ok=True)

    pairs = pd.DataFrame(
        [
            {
                "pair_id": 1,
                "label": 0,
                "split": "val",
                "subject_a": 23,
                "subject_b": 182,
                "frgp": 0,
                "path_a": "C:/fingerprint-research/data/raw/PolyU_Hong_Kong/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images/first_session/p23/p1.bmp",
                "path_b": "C:/fingerprint-research/data/raw/PolyU_Hong_Kong/Cross_Fingerprint_Images_Database/contact-based_fingerprints/first_session/182_6.jpg",
            }
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "label": 0,
                "split": "val",
                "path_a": "C:/fingerprint-research/data/raw/polyu_cross/processed_contactless_2d_fingerprint_images/first_session/p24/p1.bmp",
                "path_b": "C:/fingerprint-research/data/raw/polyu_cross/contact-based_fingerprints/first_session/332_4.jpg",
                "score": 0.51,
            }
        ]
    )
    scores.to_csv(run_root / "scores_dl_quick_val.csv", index=False)

    monkeypatch.setattr("fpbench.catalog.demo_catalog.BENCH_ROOT", bench_root)
    bundle = DatasetBundle(
        dataset="polyu_cross",
        manifest=pd.DataFrame(),
        pairs_by_split={"val": pairs},
        stats={},
        split_meta={},
        protocol_note_path=None,
        manifest_lookup={},
        manifest_lookup_by_signature={},
        benchmark_best={"polyu_cross:val": {"best_auc": {"method": "dl_quick", "run": "full_polyu_cross_h5"}}},
    )

    selection = select_case_row(
        bundle,
        {
            "dataset": "polyu_cross",
            "split": "val",
            "case_type": "cross_modality_impostor",
            "label": 0,
            "difficulty": "hard",
            "selector": "benchmark_top",
        },
    )

    assert selection is not None
    assert selection.benchmark_context["selection_driver"] == "heuristic_fallback"
    assert selection.benchmark_context["benchmark_fallback_reason"].startswith("Score file scores_dl_quick_val.csv had no overlapping pair keys")
    assert "heuristic impostor fallback" in selection.selection_policy
    assert selection.selection_diagnostics["fallback_category"] == "score_file_no_pair_overlap"
    assert selection.selection_diagnostics["heuristic_fallback_used"] is True


def test_build_catalog_bundle_discovers_benchmark_best_without_best_methods_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _configure_temp_catalog_roots(monkeypatch, repo_root)
    _stub_catalog_postprocessing(monkeypatch)
    _patch_minimal_catalog_config(monkeypatch, dataset="nist_sd300b")
    paths = _write_minimal_nist_catalog_fixture(repo_root, dataset="nist_sd300b")

    run_root = repo_root / "artifacts" / "reports" / "benchmark" / "current"
    run_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-04-17T00:00:00Z",
                "method": "sift",
                "split": "val",
                "n_pairs": 1,
                "auc": 0.91,
                "eer": 0.09,
                "avg_ms_pair_wall": 12.5,
                "config_json": json.dumps(
                    {
                        "dataset": "nist_sd300b",
                        "method_semantics_epoch": "sift_runtime_aligned_v1",
                    }
                ),
            }
        ]
    ).to_csv(run_root / "results_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "label": 1,
                "split": "val",
                "path_a": paths["plain_subject_1"],
                "path_b": paths["roll_subject_1"],
                "score": 0.91,
            }
        ]
    ).to_csv(run_root / "scores_sift_val.csv", index=False)
    (run_root / "validation.ok").write_text("", encoding="utf-8")

    bundle = build_catalog_bundle(write_files=False, generated_at="2026-04-17T00:00:00Z")
    catalog = bundle["catalog"]

    assert len(catalog["verify_cases"]) == 1
    case = catalog["verify_cases"][0]
    assert case["recommended_method"] == "sift"
    assert case["selection_policy"].startswith("benchmark_top via current/sift")
    assert case["benchmark_context"]["selection_driver"] == "benchmark_driven"
    assert case["benchmark_context"]["benchmark_run"] == "current"
    assert case["benchmark_context"]["benchmark_method"] == "sift"
    assert case["benchmark_context"]["benchmark_score"] == 0.91
    assert case["benchmark_context"]["artifact_source"] == "scores_sift_val.csv"
    assert case["benchmark_context"]["benchmark_best_source"] == "results_summary_scan"
    assert case["selection_diagnostics"]["benchmark_discovery_outcome"] == "benchmark_best_resolved"
    assert case["selection_diagnostics"]["benchmark_best_source"] == "results_summary_scan"
    assert case["selection_diagnostics"]["selection_driver"] == "benchmark_driven"
    assert case["selection_diagnostics"]["score_artifact_path_used"] == "artifacts/reports/benchmark/current/scores_sift_val.csv"

    assert catalog["source_datasets"][0]["benchmark_runs"] == [
        {
            "split": "val",
            "run": "current",
            "best_auc_method": "sift",
            "benchmark_best_source": "results_summary_scan",
        }
    ]
    dataset_summary = catalog["source_datasets"][0]["verify_selection_diagnostics"][0]
    assert dataset_summary["benchmark_best_source"] == "results_summary_scan"
    assert dataset_summary["benchmark_backed_selection_succeeded"] is True
    assert dataset_summary["heuristic_fallback_used"] is False


def test_discover_benchmark_best_from_artifacts_ignores_legacy_sift_rows_even_if_they_score_better(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _configure_temp_catalog_roots(monkeypatch, repo_root)

    run_root = repo_root / "artifacts" / "reports" / "benchmark" / "current"
    run_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-04-17T00:00:00Z",
                "method": "sift",
                "split": "val",
                "n_pairs": 1,
                "auc": 0.99,
                "eer": 0.01,
                "avg_ms_pair_wall": 1.0,
                "config_json": json.dumps({"dataset": "nist_sd300b"}),
            },
            {
                "timestamp_utc": "2026-04-18T00:00:00Z",
                "method": "sift",
                "split": "val",
                "n_pairs": 1,
                "auc": 0.91,
                "eer": 0.09,
                "avg_ms_pair_wall": 12.5,
                "config_json": json.dumps(
                    {
                        "dataset": "nist_sd300b",
                        "method_semantics_epoch": "sift_runtime_aligned_v1",
                    }
                ),
            },
        ]
    ).to_csv(run_root / "results_summary.csv", index=False)
    (run_root / "scores_sift_val.csv").write_text("label,score\n1,0.91\n", encoding="utf-8")
    (run_root / "validation.ok").write_text("", encoding="utf-8")

    discovered = demo_catalog.discover_benchmark_best_from_artifacts()

    assert discovered["nist_sd300b:val"]["best_auc"]["method"] == "sift"
    assert discovered["nist_sd300b:val"]["best_auc"]["auc"] == 0.91
    assert discovered["nist_sd300b:val"]["best_auc"]["method_semantics_epoch"] == "sift_runtime_aligned_v1"
    assert discovered["nist_sd300b:val"]["best_eer"]["eer"] == 0.09


def test_load_benchmark_best_ignores_legacy_best_methods_json_entries_for_sift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _configure_temp_catalog_roots(monkeypatch, repo_root)

    run_root = repo_root / "artifacts" / "reports" / "benchmark" / "current"
    run_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-04-18T00:00:00Z",
                "method": "sift",
                "split": "val",
                "n_pairs": 1,
                "auc": 0.91,
                "eer": 0.09,
                "avg_ms_pair_wall": 12.5,
                "config_json": json.dumps(
                    {
                        "dataset": "nist_sd300b",
                        "method_semantics_epoch": "sift_runtime_aligned_v1",
                    }
                ),
            },
        ]
    ).to_csv(run_root / "results_summary.csv", index=False)
    (run_root / "scores_sift_val.csv").write_text("label,score\n1,0.91\n", encoding="utf-8")
    (run_root / "validation.ok").write_text("", encoding="utf-8")
    _write_json(
        repo_root / "artifacts" / "reports" / "benchmark" / "april_comparison" / "best_methods.json",
        {
            "nist_sd300b:val": {
                "benchmark_best_source": "best_methods_json",
                "best_auc": {
                    "method": "sift",
                    "run": "legacy_snapshot",
                    "auc": 0.99,
                },
            }
        },
    )

    best = demo_catalog.load_benchmark_best()

    assert best["nist_sd300b:val"]["best_auc"]["run"] == "current"
    assert best["nist_sd300b:val"]["best_auc"]["auc"] == 0.91
    assert best["nist_sd300b:val"]["best_auc"]["method_semantics_epoch"] == "sift_runtime_aligned_v1"


def test_build_catalog_bundle_canonicalizes_classic_v2_benchmark_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _configure_temp_catalog_roots(monkeypatch, repo_root)
    _stub_catalog_postprocessing(monkeypatch)
    _patch_minimal_catalog_config(monkeypatch, dataset="nist_sd300b")
    paths = _write_minimal_nist_catalog_fixture(repo_root, dataset="nist_sd300b")

    run_root = repo_root / "artifacts" / "reports" / "benchmark" / "current"
    run_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-04-17T00:00:00Z",
                "method": "classic_v2",
                "split": "val",
                "n_pairs": 1,
                "auc": 0.88,
                "eer": 0.12,
                "avg_ms_pair_wall": 19.0,
                "config_json": json.dumps({"dataset": "nist_sd300b"}),
            }
        ]
    ).to_csv(run_root / "results_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "label": 1,
                "split": "val",
                "path_a": paths["plain_subject_1"],
                "path_b": paths["roll_subject_1"],
                "score": 0.88,
            }
        ]
    ).to_csv(run_root / "scores_classic_v2_val.csv", index=False)
    (run_root / "validation.ok").write_text("", encoding="utf-8")

    bundle = build_catalog_bundle(write_files=False, generated_at="2026-04-17T00:00:00Z")
    catalog = bundle["catalog"]

    assert len(catalog["verify_cases"]) == 1
    case = catalog["verify_cases"][0]
    assert case["recommended_method"] == "classic_gftt_orb"
    assert case["selection_policy"].startswith("benchmark_top via current/classic_v2")
    assert case["benchmark_context"]["benchmark_method"] == "classic_v2"
    assert case["benchmark_context"]["benchmark_score"] == 0.88
    assert case["benchmark_context"]["artifact_source"] == "scores_classic_v2_val.csv"
    assert case["selection_diagnostics"]["chosen_raw_benchmark_method"] == "classic_v2"
    assert case["selection_diagnostics"]["chosen_canonical_method"] == "classic_gftt_orb"


def test_build_catalog_bundle_uses_heuristics_when_no_benchmark_run_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _configure_temp_catalog_roots(monkeypatch, repo_root)
    _stub_catalog_postprocessing(monkeypatch)
    _patch_minimal_catalog_config(monkeypatch, dataset="nist_sd300b")
    _write_minimal_nist_catalog_fixture(repo_root, dataset="nist_sd300b")

    bundle = build_catalog_bundle(write_files=False, generated_at="2026-04-17T00:00:00Z")
    catalog = bundle["catalog"]

    assert len(catalog["verify_cases"]) == 1
    case = catalog["verify_cases"][0]
    assert case["recommended_method"] == "sift"
    assert "heuristic challenge fallback" in case["selection_policy"]
    assert case["benchmark_context"]["selection_driver"] == "heuristic_fallback"
    assert case["benchmark_context"]["benchmark_run"] is None
    assert case["benchmark_context"]["benchmark_method"] is None
    assert case["benchmark_context"]["benchmark_fallback_reason"].startswith(
        "No benchmark run and raw method could be resolved for dataset nist_sd300b split val."
    )
    assert case["benchmark_context"]["source"] == "dataset_fallback"
    assert case["selection_diagnostics"]["benchmark_discovery_outcome"] == "dataset_fallback_no_benchmark_evidence"
    assert case["selection_diagnostics"]["fallback_category"] == "benchmark_resolution_missing"
    assert case["selection_diagnostics"]["fallback_reason"].startswith(
        "No benchmark run and raw method could be resolved for dataset nist_sd300b split val."
    )
    assert case["selection_diagnostics"]["chosen_canonical_method"] == "sift"
    assert catalog["source_datasets"][0]["benchmark_runs"] == []
    dataset_summary = catalog["source_datasets"][0]["verify_selection_diagnostics"][0]
    assert dataset_summary["benchmark_best_available"] is False
    assert dataset_summary["heuristic_fallback_used"] is True
    assert dataset_summary["curated_cases_built_successfully"] == 1
    assert any("No benchmark-best evidence was available" in note for note in dataset_summary["notes"])


def test_select_case_row_benchmark_selector_no_longer_returns_none_when_run_discovery_fails() -> None:
    pairs = pd.DataFrame(
        [
            {
                "pair_id": 1,
                "label": 1,
                "split": "val",
                "subject_a": 1,
                "subject_b": 1,
                "frgp": 1,
                "path_a": "C:/fixture/nist_sd300b/subject1_plain.png",
                "path_b": "C:/fixture/nist_sd300b/subject1_roll.png",
            }
        ]
    )
    bundle = DatasetBundle(
        dataset="nist_sd300b",
        manifest=pd.DataFrame(),
        pairs_by_split={"val": pairs},
        stats={},
        split_meta={},
        protocol_note_path=None,
        manifest_lookup={},
        manifest_lookup_by_signature={},
        benchmark_best={},
    )

    selection = select_case_row(
        bundle,
        {
            "dataset": "nist_sd300b",
            "split": "val",
            "case_type": "easy_genuine",
            "label": 1,
            "difficulty": "easy",
            "selector": "benchmark_bottom",
        },
    )

    assert selection is not None
    assert selection.benchmark_context["selection_driver"] == "heuristic_fallback"
    assert selection.benchmark_context["benchmark_run"] is None
    assert selection.benchmark_context["benchmark_method"] is None
    assert selection.benchmark_context["benchmark_fallback_reason"].startswith(
        "No benchmark run and raw method could be resolved for dataset nist_sd300b split val."
    )
    assert "heuristic challenge fallback" in selection.selection_policy
    assert selection.selection_diagnostics["benchmark_discovery_outcome"] == "dataset_fallback_no_benchmark_evidence"
    assert selection.selection_diagnostics["fallback_category"] == "benchmark_resolution_missing"


def test_build_catalog_bundle_report_surfaces_dataset_selection_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _configure_temp_catalog_roots(monkeypatch, repo_root)
    _patch_minimal_catalog_config(monkeypatch, dataset="nist_sd300b")
    _write_minimal_nist_catalog_fixture(repo_root, dataset="nist_sd300b")

    bundle = build_catalog_bundle(write_files=False, generated_at="2026-04-17T00:00:00Z")
    report = bundle["report"]
    catalog = bundle["catalog"]

    assert report["validation_status"] == "fail"
    assert report["validation_warnings_count"] > 0
    assert report["selection_diagnostics"]["catalog_build_health"]["status"] == "degraded"
    assert report["selection_diagnostics"]["catalog_build_health"]["datasets_with_missing_benchmark_evidence"] == ["nist_sd300b:val"]

    dataset_summary = report["selection_diagnostics"]["dataset_verify_selection"][0]
    assert dataset_summary["dataset"] == "nist_sd300b"
    assert dataset_summary["split"] == "val"
    assert dataset_summary["benchmark_best_available"] is False
    assert dataset_summary["heuristic_fallback_used"] is True
    assert dataset_summary["curated_cases_built_successfully"] == 1
    assert any("No benchmark-best evidence was available" in note for note in dataset_summary["notes"])

    case_summary = report["selection_diagnostics"]["case_selection"][0]
    assert case_summary["case_id"] == "fixture_verify_case"
    assert case_summary["selection_driver"] == "heuristic_fallback"
    assert case_summary["chosen_canonical_method"] == "sift"
    assert catalog["metadata"]["catalog_build_health"]["status"] == "degraded"


def test_demo_catalog_materializes_binary_png_assets_and_thumbnails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _configure_temp_catalog_roots(monkeypatch, repo_root)

    source_path = repo_root / "data" / "raw" / "demo_ds" / "sample.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((48, 36), 180, dtype=np.uint8)
    image[:, 12:24] = 40
    assert cv2.imwrite(str(source_path), image)

    asset = materialize_asset_dict(
        {
            "asset_id": "asset_demo_png",
            "dataset": "demo_ds",
            "path": str(source_path),
            "source_path": str(source_path),
            "source_relative_path": source_path.relative_to(repo_root).as_posix(),
            "signature": "demo/sample.png",
            "subject_id": 1,
            "capture": "plain",
            "finger": "1",
            "split": "val",
            "availability_status": "available",
            "traceability": {"source_dataset": "demo_ds"},
        }
    )

    materialized_path = repo_root / asset["relative_path"]
    thumbnail_path = repo_root / asset["thumbnail_path"]
    assert asset["materialized_asset_kind"] == MATERIALIZED_ASSET_KIND
    assert materialized_path.suffix.lower() == ".png"
    assert thumbnail_path.suffix.lower() == ".png"
    assert materialized_path.is_file()
    assert thumbnail_path.is_file()
    assert materialized_path.name != "asset_demo_png.json"
    assert asset["availability_status"] == "available"
    assert asset["content_type"] == "image/png"
    assert asset["dimensions"] == {"width": 36, "height": 48}
    assert asset["thumbnail_dimensions"]["height"] <= 48
    assert cv2.imread(str(materialized_path), cv2.IMREAD_UNCHANGED) is not None
    assert cv2.imread(str(thumbnail_path), cv2.IMREAD_UNCHANGED) is not None


def test_demo_catalog_materializes_polyu_3d_surface_mat_to_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _configure_temp_catalog_roots(monkeypatch, repo_root)

    source_path = repo_root / "data" / "raw" / "polyu_3d" / "001_1.mat"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    surface = np.array(
        [
            [0.0, 1.0, 2.0, np.nan],
            [0.5, 1.5, 2.5, 3.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    scipy_io.savemat(source_path, {"surface": surface})

    asset = materialize_asset_dict(
        {
            "asset_id": "asset_polyu_surface",
            "dataset": "polyu_3d",
            "path": str(source_path),
            "source_path": str(source_path),
            "source_relative_path": source_path.relative_to(repo_root).as_posix(),
            "signature": "polyu_3d/001_1.mat",
            "subject_id": 1,
            "capture": "contactless",
            "finger": "unknown",
            "split": "val",
            "session": 1,
            "source_modality": "contactless_3d_surface",
            "availability_status": "available",
            "traceability": {"source_dataset": "polyu_3d"},
        }
    )

    materialized_path = repo_root / asset["relative_path"]
    thumbnail_path = repo_root / asset["thumbnail_path"]
    rendered = cv2.imread(str(materialized_path), cv2.IMREAD_UNCHANGED)
    thumb = cv2.imread(str(thumbnail_path), cv2.IMREAD_UNCHANGED)

    assert asset["availability_status"] == "available"
    assert asset["materialized_asset_kind"] == MATERIALIZED_ASSET_KIND
    assert asset["source_modality"] == "contactless_3d_surface"
    assert materialized_path.is_file()
    assert thumbnail_path.is_file()
    assert rendered is not None
    assert thumb is not None
    assert rendered.dtype == np.uint8
    assert rendered.ndim == 2
    assert thumb.dtype == np.uint8
    assert thumb.ndim == 2
    assert rendered.shape[:2] == (3, 4)
    assert thumb.shape[0] <= rendered.shape[0]
    assert thumb.shape[1] <= rendered.shape[1]


def test_demo_catalog_polyu_3d_never_recommends_photometric_raw() -> None:
    manifest = pd.DataFrame(
        [
            {
                "dataset": "polyu_3d",
                "capture": "contactless",
                "subject_id": 7,
                "impression": "sample_01",
                "ppi": 0,
                "frgp": 0,
                "path": "C:/virtual/raw_frame.bmp",
                "split": "val",
                "sample_id": 1,
                "session": 1,
                "source_modality": "contactless_2d_photometric_raw",
            },
            {
                "dataset": "polyu_3d",
                "capture": "contactless",
                "subject_id": 7,
                "impression": "sample_01",
                "ppi": 0,
                "frgp": 0,
                "path": "C:/virtual/001_1_first.mat",
                "split": "val",
                "sample_id": 1,
                "session": 1,
                "source_modality": "contactless_3d_surface",
            },
            {
                "dataset": "polyu_3d",
                "capture": "contactless",
                "subject_id": 7,
                "impression": "sample_01",
                "ppi": 0,
                "frgp": 0,
                "path": "C:/virtual/001_1_second.mat",
                "split": "val",
                "sample_id": 1,
                "session": 2,
                "source_modality": "contactless_3d_surface",
            },
        ]
    )
    bundle = DatasetBundle(
        dataset="polyu_3d",
        manifest=manifest,
        pairs_by_split={},
        stats={},
        split_meta={},
        protocol_note_path=None,
        manifest_lookup={},
        manifest_lookup_by_signature={},
        benchmark_best={},
    )

    identity = build_identity_record(bundle)
    enrollment = find_asset(identity, identity.recommended_enrollment_asset_id)
    probe = find_asset(identity, identity.recommended_probe_asset_id)

    assert all(asset.source_modality == "contactless_3d_surface" for asset in identity.exemplars)
    assert enrollment.source_modality == "contactless_3d_surface"
    assert enrollment.session == 1
    assert probe.source_modality == "contactless_3d_surface"
    assert probe.session == 2


def test_demo_catalog_unsw_uses_optical2d_to_reconstructed3d_for_identity_flow() -> None:
    manifest = pd.DataFrame(
        [
            {
                "dataset": "unsw_2d3d",
                "capture": "contact_based",
                "subject_id": 11,
                "impression": "capture_01",
                "ppi": 0,
                "frgp": 2,
                "path": "C:/virtual/11_2_1.bmp",
                "split": "val",
                "sample_id": 1,
                "session": 1,
                "source_modality": "optical_2d",
            },
            {
                "dataset": "unsw_2d3d",
                "capture": "contactless",
                "subject_id": 11,
                "impression": "capture_01",
                "ppi": 0,
                "frgp": 2,
                "path": "C:/virtual/SIRE-11_2_1.bmp",
                "split": "val",
                "sample_id": 1,
                "session": 1,
                "source_modality": "reconstructed_3d",
            },
            {
                "dataset": "unsw_2d3d",
                "capture": "contactless",
                "subject_id": 11,
                "impression": "capture_01",
                "ppi": 0,
                "frgp": 2,
                "path": "C:/virtual/SIRE-11_2_1_HT1.bmp",
                "split": "val",
                "sample_id": 1,
                "session": 1,
                "source_modality": "derived_3d_variant",
            },
            {
                "dataset": "unsw_2d3d",
                "capture": "contactless",
                "subject_id": 11,
                "impression": "capture_01",
                "ppi": 0,
                "frgp": 2,
                "path": "C:/virtual/raw/11_2_1_0.bmp",
                "split": "val",
                "sample_id": 1,
                "session": 1,
                "source_modality": "reconstruction_intermediate",
            },
            {
                "dataset": "unsw_2d3d",
                "capture": "contact_based",
                "subject_id": 12,
                "impression": "capture_01",
                "ppi": 0,
                "frgp": 2,
                "path": "C:/virtual/12_2_1.bmp",
                "split": "val",
                "sample_id": 1,
                "session": 1,
                "source_modality": "optical_2d",
            },
            {
                "dataset": "unsw_2d3d",
                "capture": "contactless",
                "subject_id": 12,
                "impression": "capture_01",
                "ppi": 0,
                "frgp": 2,
                "path": "C:/virtual/SIRE-12_2_1.bmp",
                "split": "val",
                "sample_id": 1,
                "session": 1,
                "source_modality": "reconstructed_3d",
            },
        ]
    )
    bundle = DatasetBundle(
        dataset="unsw_2d3d",
        manifest=manifest,
        pairs_by_split={},
        stats={},
        split_meta={},
        protocol_note_path=None,
        manifest_lookup={},
        manifest_lookup_by_signature={},
        benchmark_best={},
    )

    identity = build_identity_record(bundle)
    enrollment = find_asset(identity, identity.recommended_enrollment_asset_id)
    probe = find_asset(identity, identity.recommended_probe_asset_id)

    assert enrollment.source_modality == "optical_2d"
    assert probe.source_modality == "reconstructed_3d"
    assert all(asset.source_modality in {"optical_2d", "reconstructed_3d"} for asset in identity.exemplars)

    scenarios = build_identification_scenarios({"unsw_2d3d": bundle}, [identity])
    assert len(scenarios) == 1
    assert scenarios[0].scenario_type == "no_match"
    assert scenarios[0].probe_asset.source_modality == "reconstructed_3d"
