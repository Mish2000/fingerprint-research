from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

import apps.api.catalog_store as catalog_store
import apps.api.demo_store as demo_store
from apps.api.main import (
    catalog_asset,
    catalog_dataset_browser,
    catalog_datasets,
    catalog_identify_gallery,
    catalog_verify_case,
    catalog_verify_cases,
    demo_cases,
)


@pytest.fixture(autouse=True)
def _clear_catalog_and_demo_caches():
    catalog_store.clear_catalog_store_cache()
    demo_store.clear_demo_store_cache()
    yield
    catalog_store.clear_catalog_store_cache()
    demo_store.clear_demo_store_cache()


def _write_file(path: Path, payload: bytes = b"asset") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _demo_asset_payload(
    asset_id: str,
    relative_path: str,
    *,
    dataset: str,
    capture: str,
    finger: str,
    recommended_usage: str,
    availability_status: str = "available",
) -> dict:
    return {
        "asset_id": asset_id,
        "dataset": dataset,
        "path": relative_path,
        "relative_path": relative_path,
        "availability_status": availability_status,
        "capture": capture,
        "finger": finger,
        "traceability": {
            "source_dataset": dataset,
            "materialized_asset_kind": "binary_image",
        },
        "recommended_usage": recommended_usage,
    }


def _verify_case_payload(
    case_id: str,
    *,
    dataset: str,
    asset_a: dict,
    asset_b: dict,
    difficulty: str = "easy",
    case_type: str = "easy_genuine",
    ground_truth: str = "match",
    split: str = "val",
    recommended_method: str = "sift",
    tags: list[str] | None = None,
    modality_relation: str = "same_modality",
    is_demo_safe: bool = True,
    availability_status: str = "available",
    benchmark_context: dict | None = None,
    selection_diagnostics: dict | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "title": f"Case {case_id}",
        "description": f"Description for {case_id}",
        "dataset": dataset,
        "split": split,
        "case_type": case_type,
        "difficulty": difficulty,
        "ground_truth": ground_truth,
        "recommended_method": recommended_method,
        "capture_a": asset_a["capture"],
        "capture_b": asset_b["capture"],
        "image_a": asset_a,
        "image_b": asset_b,
        "is_demo_safe": is_demo_safe,
        "availability_status": availability_status,
        "selection_reason": f"Selected for {case_id}.",
        "selection_policy": "catalog_priority",
        "tags": tags or [],
        "modality_relation": modality_relation,
        "benchmark_context": benchmark_context or {
            "run": "legacy_reference",
            "method": "sift",
        },
        "selection_diagnostics": selection_diagnostics or {
            "selection_driver": "benchmark_driven",
            "benchmark_discovery_outcome": "benchmark_best_resolved",
            "benchmark_selection_status": "benchmark_score_used",
            "benchmark_backed_selection": True,
            "heuristic_fallback_used": False,
        },
    }


def _identity_payload(
    identity_id: str,
    *,
    dataset: str,
    display_name: str,
    subject_id: int,
    exemplars: list[dict],
    enrollment_candidates: list[str],
    probe_candidates: list[str],
    is_demo_safe: bool = True,
) -> dict:
    return {
        "identity_id": identity_id,
        "dataset": dataset,
        "display_name": display_name,
        "subject_id": subject_id,
        "gallery_role": "standard",
        "tags": [dataset, "gallery"],
        "is_demo_safe": is_demo_safe,
        "enrollment_candidates": enrollment_candidates,
        "probe_candidates": probe_candidates,
        "exemplars": exemplars,
    }


def _browser_item_payload(
    asset_id: str,
    *,
    dataset: str,
    thumbnail_path: str,
    preview_path: str,
    split: str,
    subject_id: str,
    finger: str,
    capture: str,
    modality: str,
    ui_eligible: bool = True,
    availability_status: str = "available",
) -> dict:
    return {
        "asset_id": asset_id,
        "dataset": dataset,
        "split": split,
        "source_path": f"data/raw/{dataset}/{asset_id}.png",
        "thumbnail_path": thumbnail_path,
        "preview_path": preview_path,
        "availability_status": availability_status,
        "subject_id": subject_id,
        "finger": finger,
        "capture": capture,
        "modality": modality,
        "traceability": {
            "manifest_path": f"data/manifests/{dataset}/manifest.csv",
            "manifest_row_number": 1,
        },
        "ui_eligible": ui_eligible,
        "selection_reason": f"Selected for browser: {asset_id}.",
        "selection_policy": "deterministic_round_robin",
        "original_dimensions": {"width": 640, "height": 480},
        "thumbnail_dimensions": {"width": 160, "height": 120},
        "preview_dimensions": {"width": 512, "height": 384},
    }


def _configure_artifact_roots(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    for module in (demo_store, catalog_store):
        monkeypatch.setattr(module, "ROOT", repo_root)
        monkeypatch.setattr(module, "SAMPLES_ROOT", repo_root / "data" / "samples")
        monkeypatch.setattr(module, "CATALOG_PATH", repo_root / "data" / "samples" / "catalog.json")
        monkeypatch.setattr(module, "ASSETS_ROOT", repo_root / "data" / "samples" / "assets")
    monkeypatch.setattr(catalog_store, "PROCESSED_ROOT", repo_root / "data" / "processed")
    monkeypatch.setattr(catalog_store, "UI_ASSETS_REGISTRY_PATH", repo_root / "data" / "processed" / "ui_assets_registry.json")
    monkeypatch.delenv("FPBENCH_DEMO_ROOT", raising=False)
    monkeypatch.delenv("FPBENCH_DEMO_CATALOG_PATH", raising=False)
    monkeypatch.delenv("FPBENCH_DEMO_ASSETS_ROOT", raising=False)
    monkeypatch.delenv("FPBENCH_UI_ASSETS_ROOT", raising=False)
    monkeypatch.delenv("FPBENCH_UI_ASSETS_REGISTRY_PATH", raising=False)
    catalog_store.clear_catalog_store_cache()
    demo_store.clear_demo_store_cache()


def _build_catalog_and_browser_artifacts(repo_root: Path) -> None:
    verify_left = _write_file(repo_root / "data" / "samples" / "assets" / "verify_demo" / "left.png", b"left")
    verify_right = _write_file(repo_root / "data" / "samples" / "assets" / "verify_demo" / "right.png", b"right")
    verify_alt = _write_file(repo_root / "data" / "samples" / "assets" / "verify_demo" / "alt.png", b"alt")
    identity_asset = _write_file(repo_root / "data" / "samples" / "assets" / "identify_demo" / "id_a.png", b"id")

    case_asset_a = _demo_asset_payload(
        "asset_verify_left",
        verify_left.relative_to(repo_root).as_posix(),
        dataset="verify_demo",
        capture="plain",
        finger="1",
        recommended_usage="verify_left",
    )
    case_asset_b = _demo_asset_payload(
        "asset_verify_right",
        verify_right.relative_to(repo_root).as_posix(),
        dataset="verify_demo",
        capture="roll",
        finger="1",
        recommended_usage="verify_right",
    )
    case_asset_c = _demo_asset_payload(
        "asset_verify_alt",
        verify_alt.relative_to(repo_root).as_posix(),
        dataset="verify_demo",
        capture="plain",
        finger="2",
        recommended_usage="verify_left",
    )
    identity_exemplar = _demo_asset_payload(
        "asset_identity_a",
        identity_asset.relative_to(repo_root).as_posix(),
        dataset="identify_demo",
        capture="contactless",
        finger="3",
        recommended_usage="recommended_enrollment",
    )

    catalog_payload = {
        "catalog_version": "1.0.0",
        "generated_at": "2026-03-31T00:00:00Z",
        "source_datasets": [
            {
                "dataset": "verify_demo",
                "dataset_label": "Verify Demo",
                "verify_selection_diagnostics": [
                    {
                        "dataset": "verify_demo",
                        "split": "val",
                        "planned_curated_cases": 2,
                        "curated_cases_built_successfully": 2,
                        "benchmark_best_available": False,
                        "benchmark_resolution_complete": False,
                        "heuristic_fallback_used": True,
                        "selection_driver_counts": {
                            "benchmark_driven": 1,
                            "heuristic_fallback": 1,
                        },
                    }
                ],
            },
            {"dataset": "identify_demo", "dataset_label": "Identify Demo"},
        ],
        "verify_cases": [
            _verify_case_payload(
                "verify_easy",
                dataset="verify_demo",
                asset_a=case_asset_a,
                asset_b=case_asset_b,
                difficulty="easy",
                tags=["hero", "qa"],
                selection_diagnostics={
                    "selection_driver": "benchmark_driven",
                    "benchmark_discovery_outcome": "benchmark_best_resolved",
                    "benchmark_selection_status": "benchmark_score_used",
                    "benchmark_backed_selection": True,
                    "heuristic_fallback_used": False,
                },
            ),
            _verify_case_payload(
                "verify_hard",
                dataset="verify_demo",
                asset_a=case_asset_c,
                asset_b=case_asset_b,
                difficulty="hard",
                case_type="hard_genuine",
                tags=["edge"],
                selection_diagnostics={
                    "selection_driver": "heuristic_fallback",
                    "benchmark_discovery_outcome": "dataset_fallback_no_benchmark_evidence",
                    "benchmark_selection_status": "benchmark_resolution_missing",
                    "benchmark_backed_selection": False,
                    "heuristic_fallback_used": True,
                    "fallback_category": "benchmark_resolution_missing",
                },
            ),
            _verify_case_payload(
                "verify_hidden",
                dataset="verify_demo",
                asset_a=case_asset_a,
                asset_b=case_asset_b,
                is_demo_safe=False,
            ),
        ],
        "identify_gallery": {
            "identities": [
                _identity_payload(
                    "identity_demo_1",
                    dataset="identify_demo",
                    display_name="Identify Subject 501",
                    subject_id=501,
                    exemplars=[identity_exemplar],
                    enrollment_candidates=["asset_identity_a"],
                    probe_candidates=["asset_identity_a"],
                )
            ],
            "demo_scenarios": [],
        },
        "metadata": {
            "total_verify_cases": 3,
            "total_identity_records": 1,
            "catalog_build_health": {
                "status": "degraded",
                "total_verify_cases_planned": 2,
                "total_verify_cases_built": 2,
                "case_selection_driver_counts": {
                    "benchmark_driven": 1,
                    "heuristic_fallback": 1,
                },
                "datasets_with_missing_benchmark_evidence": ["verify_demo:val"],
            },
        },
    }
    _write_json(repo_root / "data" / "samples" / "catalog.json", catalog_payload)

    thumb_good = _write_file(
        repo_root / "data" / "processed" / "browser_demo" / "ui_assets" / "thumbnails" / "browser_good.png",
        b"thumb-good",
    )
    preview_good = _write_file(
        repo_root / "data" / "processed" / "browser_demo" / "ui_assets" / "previews" / "browser_good.png",
        b"preview-good",
    )
    _write_file(
        repo_root / "data" / "processed" / "browser_demo" / "ui_assets" / "thumbnails" / "browser_missing.png",
        b"thumb-missing",
    )

    browser_index = {
        "dataset": "browser_demo",
        "dataset_label": "Browser Demo",
        "generated_at": "2026-03-31T01:00:00Z",
        "generator_version": "1.0.0",
        "selection_policy": "deterministic_round_robin",
        "validation_status": "pass_with_warnings",
        "summary": {
            "source_records_checked": 10,
            "items_generated": 3,
        },
        "items": [
            _browser_item_payload(
                "browser_good",
                dataset="browser_demo",
                thumbnail_path=thumb_good.relative_to(repo_root).as_posix(),
                preview_path=preview_good.relative_to(repo_root).as_posix(),
                split="test",
                subject_id="200",
                finger="2",
                capture="contactless",
                modality="optical_2d",
            ),
            _browser_item_payload(
                "browser_missing_preview",
                dataset="browser_demo",
                thumbnail_path="data/processed/browser_demo/ui_assets/thumbnails/browser_missing.png",
                preview_path="data/processed/browser_demo/ui_assets/previews/browser_missing.png",
                split="val",
                subject_id="300",
                finger="1",
                capture="plain",
                modality="optical_2d",
            ),
            _browser_item_payload(
                "browser_unsafe_path",
                dataset="browser_demo",
                thumbnail_path="data/processed/browser_demo/ui_assets/thumbnails/browser_good.png",
                preview_path="../escape.png",
                split="train",
                subject_id="400",
                finger="4",
                capture="plain",
                modality="latent",
            ),
        ],
    }
    browser_report = {
        "dataset": "browser_demo",
        "dataset_label": "Browser Demo",
        "generated_at": "2026-03-31T01:00:00Z",
        "generator_version": "1.0.0",
        "selection_policy": "deterministic_round_robin",
        "source_records_checked": 10,
        "generated_items": 3,
        "excluded_records": 2,
        "missing_source_files": 0,
        "unreadable_source_files": 0,
        "missing_critical_metadata": 0,
        "duplicates_skipped": 0,
        "validation_status": "pass_with_warnings",
    }
    _write_json(repo_root / "data" / "processed" / "browser_demo" / "ui_assets" / "index.json", browser_index)
    _write_json(
        repo_root / "data" / "processed" / "browser_demo" / "ui_assets" / "validation_report.json",
        browser_report,
    )
    _write_json(
        repo_root / "data" / "processed" / "ui_assets_registry.json",
        {
            "generated_at": "2026-03-31T01:05:00Z",
            "generator_version": "1.0.0",
            "datasets": [
                {
                    "dataset": "browser_demo",
                    "index_path": "data/processed/browser_demo/ui_assets/index.json",
                    "validation_report_path": "data/processed/browser_demo/ui_assets/validation_report.json",
                    "item_count": 3,
                    "validation_status": "pass_with_warnings",
                    "selection_policy": "deterministic_round_robin",
                    "summary": {
                        "items_generated": 3,
                    },
                }
            ],
        },
    )


def test_catalog_datasets_combines_catalog_and_registry_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_catalog_and_browser_artifacts(repo_root)
    _configure_artifact_roots(monkeypatch, repo_root)

    response = catalog_datasets()

    assert [item.dataset for item in response.items] == ["browser_demo", "identify_demo", "verify_demo"]
    browser_demo = next(item for item in response.items if item.dataset == "browser_demo")
    verify_demo = next(item for item in response.items if item.dataset == "verify_demo")
    identify_demo = next(item for item in response.items if item.dataset == "identify_demo")

    assert browser_demo.has_browser_assets is True
    assert browser_demo.has_verify_cases is False
    assert browser_demo.browser_item_count == 3
    assert browser_demo.available_features == ["dataset_browser"]

    assert verify_demo.has_verify_cases is True
    assert verify_demo.verify_case_count == 2
    assert verify_demo.has_browser_assets is False
    assert verify_demo.demo_health is not None
    assert verify_demo.demo_health.status == "degraded"
    assert verify_demo.demo_health.planned_verify_cases == 2
    assert verify_demo.demo_health.benchmark_backed_cases == 1
    assert verify_demo.demo_health.heuristic_fallback_cases == 1

    assert identify_demo.has_identify_gallery is True
    assert identify_demo.identify_identity_count == 1
    assert response.catalog_build_health is not None
    assert response.catalog_build_health.catalog_build_status == "degraded"
    assert response.catalog_build_health.total_verify_cases == 2
    assert response.catalog_build_health.heuristic_fallback_case_count == 1


def test_catalog_verify_cases_filters_and_paginates_and_preserves_demo_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_catalog_and_browser_artifacts(repo_root)
    _configure_artifact_roots(monkeypatch, repo_root)

    response = catalog_verify_cases(dataset="verify_demo", difficulty="hard", tag="edge", limit=1, offset=0)

    assert response.total == 1
    assert response.limit == 1
    assert response.offset == 0
    assert response.has_more is False
    assert [item.case_id for item in response.items] == ["verify_hard"]
    assert response.items[0].image_a_url == "/api/demo/cases/verify_hard/a"
    assert response.items[0].evidence_quality is not None
    assert response.items[0].evidence_quality.selection_driver == "heuristic_fallback"
    assert response.items[0].evidence_quality.evidence_status == "degraded"
    assert response.catalog_build_health is not None
    assert response.catalog_build_health.catalog_build_status == "degraded"

    demo_response = demo_cases()
    assert [item.id for item in demo_response.cases] == ["verify_easy", "verify_hard"]
    assert demo_response.cases[0].evidence_quality is not None
    assert demo_response.cases[0].evidence_quality.evidence_status == "strong"
    assert demo_response.catalog_build_health is not None
    assert demo_response.catalog_build_health.heuristic_fallback_case_count == 1


def test_catalog_verify_case_detail_returns_detail_and_missing_case_404(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_catalog_and_browser_artifacts(repo_root)
    _configure_artifact_roots(monkeypatch, repo_root)

    detail = catalog_verify_case("verify_easy")

    assert detail.case_id == "verify_easy"
    assert detail.benchmark_context["method"] == "sift"
    assert detail.benchmark_context["canonical_method"] == "sift"
    assert detail.benchmark_context["benchmark_method"] == "sift"
    assert detail.traceability_summary["asset_a_id"] == "asset_verify_left"
    assert detail.evidence_quality is not None
    assert detail.evidence_quality.evidence_status == "strong"

    with pytest.raises(HTTPException) as exc_info:
        catalog_verify_case("missing_case")

    assert exc_info.value.status_code == 404


def test_catalog_verify_case_detail_canonicalizes_benchmark_context_and_demo_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    verify_left = _write_file(repo_root / "data" / "samples" / "assets" / "verify_demo" / "left.png", b"left")
    verify_right = _write_file(repo_root / "data" / "samples" / "assets" / "verify_demo" / "right.png", b"right")

    case_asset_a = _demo_asset_payload(
        "asset_verify_left",
        verify_left.relative_to(repo_root).as_posix(),
        dataset="verify_demo",
        capture="plain",
        finger="1",
        recommended_usage="verify_left",
    )
    case_asset_b = _demo_asset_payload(
        "asset_verify_right",
        verify_right.relative_to(repo_root).as_posix(),
        dataset="verify_demo",
        capture="roll",
        finger="1",
        recommended_usage="verify_right",
    )

    _write_json(
        repo_root / "data" / "samples" / "catalog.json",
        {
            "catalog_version": "1.0.0",
            "generated_at": "2026-03-31T00:00:00Z",
            "source_datasets": [
                {"dataset": "verify_demo", "dataset_label": "Verify Demo"},
            ],
            "verify_cases": [
                _verify_case_payload(
                    "verify_alias_case",
                    dataset="verify_demo",
                    asset_a=case_asset_a,
                    asset_b=case_asset_b,
                    recommended_method="dl_quick",
                    benchmark_context={
                        "source": "benchmark_best_auc",
                        "run": "full_verify_demo_h1",
                        "method": "dl_quick",
                    },
                )
            ],
            "identify_gallery": {"identities": [], "demo_scenarios": []},
        },
    )
    _write_json(repo_root / "data" / "processed" / "ui_assets_registry.json", {"datasets": []})
    _configure_artifact_roots(monkeypatch, repo_root)

    detail = catalog_verify_case("verify_alias_case")
    demo_response = demo_cases()

    assert detail.recommended_method.value == "dl"
    assert detail.benchmark_context["method"] == "dl"
    assert detail.benchmark_context["canonical_method"] == "dl"
    assert detail.benchmark_context["benchmark_method"] == "dl_quick"
    assert detail.benchmark_context["method_label"] == "Deep Learning (ResNet50)"
    assert detail.benchmark_context["benchmark_run"] == "full_verify_demo_h1"
    assert detail.benchmark_context["artifact_source"] == "scores_dl_quick_val.csv"
    assert demo_response.cases[0].recommended_method.value == "dl"
    assert demo_response.cases[0].benchmark_method == "dl_quick"


def test_catalog_identify_gallery_returns_candidate_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_catalog_and_browser_artifacts(repo_root)
    _configure_artifact_roots(monkeypatch, repo_root)

    response = catalog_identify_gallery(dataset="identify_demo", limit=10, offset=0)

    assert response.total == 1
    identity = response.items[0]
    assert identity.identity_id == "identity_demo_1"
    assert identity.subject_id == "501"
    assert identity.enrollment_candidates[0].asset_id == "asset_identity_a"
    assert identity.enrollment_candidates[0].has_servable_asset is True


def test_catalog_dataset_browser_uses_ui_assets_index_only_with_filters_and_pagination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_catalog_and_browser_artifacts(repo_root)
    _configure_artifact_roots(monkeypatch, repo_root)

    response = catalog_dataset_browser(
        dataset="browser_demo",
        capture="contactless",
        limit=1,
        offset=0,
        sort="split_subject_asset",
    )

    assert response.dataset == "browser_demo"
    assert response.validation_status == "pass_with_warnings"
    assert response.warning_count == 2
    assert response.total == 1
    assert response.items[0].asset_id == "browser_good"
    assert response.items[0].thumbnail_url == "/api/catalog/assets/browser_demo/browser_good/thumbnail"
    assert not (repo_root / "data" / "raw").exists()
    assert not (repo_root / "data" / "manifests").exists()
    assert not (repo_root / "artifacts" / "reports" / "benchmark").exists()


def test_catalog_asset_serving_handles_valid_missing_invalid_and_unsafe_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_catalog_and_browser_artifacts(repo_root)
    _configure_artifact_roots(monkeypatch, repo_root)

    response = catalog_asset("browser_demo", "browser_good", "thumbnail")
    assert isinstance(response, FileResponse)
    assert Path(response.path).resolve() == (
        repo_root / "data" / "processed" / "browser_demo" / "ui_assets" / "thumbnails" / "browser_good.png"
    ).resolve()

    with pytest.raises(HTTPException) as invalid_variant:
        catalog_asset("browser_demo", "browser_good", "original")
    assert invalid_variant.value.status_code == 400

    with pytest.raises(HTTPException) as missing_asset:
        catalog_asset("browser_demo", "browser_missing_preview", "preview")
    assert missing_asset.value.status_code == 404

    with pytest.raises(HTTPException) as unsafe_asset:
        catalog_asset("browser_demo", "browser_unsafe_path", "preview")
    assert unsafe_asset.value.status_code == 404

    with pytest.raises(HTTPException) as unknown_dataset:
        catalog_asset("unknown_dataset", "browser_good", "thumbnail")
    assert unknown_dataset.value.status_code == 404

    with pytest.raises(HTTPException) as unknown_asset:
        catalog_asset("browser_demo", "does_not_exist", "thumbnail")
    assert unknown_asset.value.status_code == 404


def test_catalog_dataset_browser_rejects_invalid_sort_and_missing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_catalog_and_browser_artifacts(repo_root)
    _configure_artifact_roots(monkeypatch, repo_root)

    with pytest.raises(HTTPException) as invalid_sort:
        catalog_dataset_browser(dataset="browser_demo", sort="filesystem")
    assert invalid_sort.value.status_code == 400

    with pytest.raises(HTTPException) as missing_dataset:
        catalog_dataset_browser(dataset="verify_demo")
    assert missing_dataset.value.status_code == 404


def test_catalog_datasets_degrades_gracefully_when_ui_assets_registry_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _build_catalog_and_browser_artifacts(repo_root)
    (repo_root / "data" / "processed" / "ui_assets_registry.json").unlink()
    _configure_artifact_roots(monkeypatch, repo_root)

    response = catalog_datasets()

    assert [item.dataset for item in response.items] == ["identify_demo", "verify_demo"]
    assert all(item.has_browser_assets is False for item in response.items)
