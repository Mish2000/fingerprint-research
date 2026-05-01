from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

import apps.api.demo_store as demo_store
from apps.api.main import demo_case_asset, demo_cases


@pytest.fixture(autouse=True)
def _clear_demo_store_cache():
    demo_store.clear_demo_store_cache()
    yield
    demo_store.clear_demo_store_cache()


def _write_binary_asset(path: Path, payload: bytes = b"demo-asset") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _asset_payload(
    asset_id: str,
    relative_path: str,
    *,
    capture: str = "plain",
    availability_status: str = "available",
    source_path: str | None = None,
) -> dict:
    return {
        "asset_id": asset_id,
        "path": relative_path,
        "relative_path": relative_path,
        "availability_status": availability_status,
        "capture": capture,
        "source_path": source_path or "C:/raw-data/should-not-be-used.png",
    }


def _case_payload(
    case_id: str,
    asset_a: dict,
    asset_b: dict,
    *,
    is_demo_safe: bool = True,
    availability_status: str = "available",
    difficulty: str = "easy",
    ground_truth: str = "match",
    recommended_method: str = "sift",
    hidden: bool = False,
    disabled: bool = False,
    benchmark_context: dict | None = None,
    selection_diagnostics: dict | None = None,
) -> dict:
    payload = {
        "case_id": case_id,
        "title": f"Case {case_id}",
        "description": f"Description for {case_id}",
        "dataset": "synthetic_demo",
        "split": "val",
        "case_type": "easy_genuine",
        "difficulty": difficulty,
        "ground_truth": ground_truth,
        "recommended_method": recommended_method,
        "capture_a": asset_a.get("capture", "plain"),
        "capture_b": asset_b.get("capture", "plain"),
        "image_a": asset_a,
        "image_b": asset_b,
        "is_demo_safe": is_demo_safe,
        "availability_status": availability_status,
        "selection_reason": "Curated once and served directly from the catalog.",
        "selection_policy": "catalog_priority",
        "benchmark_context": benchmark_context or {
            "benchmark_run": "legacy_reference_run",
            "benchmark_method": "sift",
            "benchmark_score": 0.991,
        },
        "selection_diagnostics": selection_diagnostics or {
            "selection_driver": "benchmark_driven",
            "benchmark_discovery_outcome": "benchmark_best_resolved",
            "benchmark_selection_status": "benchmark_score_used",
            "benchmark_backed_selection": True,
            "heuristic_fallback_used": False,
        },
        "tags": ["qa", "demo"],
        "modality_relation": "same_modality",
        "hidden": hidden,
        "disabled": disabled,
    }
    return payload


def _write_catalog(
    repo_root: Path,
    verify_cases: list[dict],
    *,
    source_datasets: list[dict] | None = None,
    metadata: dict | None = None,
) -> Path:
    catalog_path = repo_root / "data" / "samples" / "catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "catalog_version": "1.0.0",
        "generated_at": "2026-03-31T00:00:00Z",
        "source_datasets": source_datasets or [
            {
                "dataset": "synthetic_demo",
                "dataset_label": "Synthetic Demo",
            }
        ],
        "verify_cases": verify_cases,
        "metadata": metadata or {},
    }
    catalog_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return catalog_path


def _configure_demo_store(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.setattr(demo_store, "ROOT", repo_root)
    monkeypatch.setattr(demo_store, "SAMPLES_ROOT", repo_root / "data" / "samples")
    monkeypatch.setattr(demo_store, "CATALOG_PATH", repo_root / "data" / "samples" / "catalog.json")
    monkeypatch.setattr(demo_store, "ASSETS_ROOT", repo_root / "data" / "samples" / "assets")
    monkeypatch.delenv("FPBENCH_DEMO_ROOT", raising=False)
    monkeypatch.delenv("FPBENCH_DEMO_CATALOG_PATH", raising=False)
    monkeypatch.delenv("FPBENCH_DEMO_ASSETS_ROOT", raising=False)
    demo_store.clear_demo_store_cache()


def test_load_demo_cases_uses_catalog_without_benchmark_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    asset_a = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "a.png")
    asset_b = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "b.png")
    _write_catalog(
        repo_root,
        [
            _case_payload(
                "catalog_only_case",
                _asset_payload("asset_a", asset_a.relative_to(repo_root).as_posix(), capture="plain"),
                _asset_payload("asset_b", asset_b.relative_to(repo_root).as_posix(), capture="roll"),
            )
        ],
    )
    _configure_demo_store(monkeypatch, repo_root)

    response = demo_store.load_demo_cases()

    assert [item.id for item in response.cases] == ["catalog_only_case"]
    assert response.cases[0].dataset_label == "Synthetic Demo"
    assert response.cases[0].image_a_url == "/api/demo/cases/catalog_only_case/a"
    assert not (repo_root / "artifacts" / "reports" / "benchmark").exists()


def test_resolve_demo_case_path_returns_both_catalog_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    asset_a = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "left.png")
    asset_b = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "right.png")
    _write_catalog(
        repo_root,
        [
            _case_payload(
                "pair_case",
                _asset_payload("left_asset", asset_a.relative_to(repo_root).as_posix(), capture="plain"),
                _asset_payload("right_asset", asset_b.relative_to(repo_root).as_posix(), capture="roll"),
            )
        ],
    )
    _configure_demo_store(monkeypatch, repo_root)

    assert demo_store.resolve_demo_case_path("pair_case", "a") == asset_a.resolve()
    assert demo_store.resolve_demo_case_path("pair_case", "b") == asset_b.resolve()


def test_missing_asset_case_is_excluded_from_demo_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    present = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "present.png")
    missing_relative = "data/samples/assets/synthetic_demo/missing.png"
    _write_catalog(
        repo_root,
        [
            _case_payload(
                "good_case",
                _asset_payload("present_a", present.relative_to(repo_root).as_posix()),
                _asset_payload("present_b", present.relative_to(repo_root).as_posix(), capture="roll"),
            ),
            _case_payload(
                "missing_asset_case",
                _asset_payload("missing_a", present.relative_to(repo_root).as_posix()),
                _asset_payload("missing_b", missing_relative),
            ),
        ],
    )
    _configure_demo_store(monkeypatch, repo_root)

    response = demo_store.load_demo_cases()

    assert [item.id for item in response.cases] == ["good_case"]
    exclusions = demo_store.get_demo_case_exclusions()
    assert "missing_asset_case" in exclusions
    assert "missing" in exclusions["missing_asset_case"].lower()


def test_invalid_case_shape_is_excluded_without_breaking_valid_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    asset_a = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "left.png")
    asset_b = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "right.png")
    valid_case = _case_payload(
        "valid_case",
        _asset_payload("asset_left", asset_a.relative_to(repo_root).as_posix()),
        _asset_payload("asset_right", asset_b.relative_to(repo_root).as_posix()),
    )
    invalid_case = _case_payload(
        "invalid_case",
        _asset_payload("asset_x", asset_a.relative_to(repo_root).as_posix()),
        _asset_payload("asset_y", asset_b.relative_to(repo_root).as_posix()),
    )
    invalid_case.pop("recommended_method")
    _write_catalog(repo_root, [valid_case, invalid_case])
    _configure_demo_store(monkeypatch, repo_root)

    response = demo_store.load_demo_cases()

    assert [item.id for item in response.cases] == ["valid_case"]
    exclusions = demo_store.get_demo_case_exclusions()
    assert "invalid_case" in exclusions
    assert "validation" in exclusions["invalid_case"].lower()


def test_path_outside_assets_root_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    safe_asset = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "ok.png")
    _write_binary_asset(repo_root / "outside.png")
    _write_catalog(
        repo_root,
        [
            _case_payload(
                "unsafe_case",
                _asset_payload("safe_asset", safe_asset.relative_to(repo_root).as_posix()),
                _asset_payload("unsafe_asset", "../outside.png"),
            )
        ],
    )
    _configure_demo_store(monkeypatch, repo_root)

    response = demo_store.load_demo_cases()

    assert response.cases == []
    exclusions = demo_store.get_demo_case_exclusions()
    assert "unsafe_case" in exclusions
    assert "path traversal" in exclusions["unsafe_case"].lower()


def test_demo_case_asset_endpoint_rejects_invalid_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    asset_a = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "a.png")
    asset_b = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "b.png")
    _write_catalog(
        repo_root,
        [
            _case_payload(
                "slot_case",
                _asset_payload("asset_a", asset_a.relative_to(repo_root).as_posix()),
                _asset_payload("asset_b", asset_b.relative_to(repo_root).as_posix()),
            )
        ],
    )
    _configure_demo_store(monkeypatch, repo_root)

    with pytest.raises(HTTPException) as exc_info:
        demo_case_asset("slot_case", "z")

    assert exc_info.value.status_code == 400
    assert "slot" in str(exc_info.value.detail).lower()


def test_demo_cases_endpoint_returns_only_available_safe_cases_and_ignores_broken_source_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    asset_a = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "first.png")
    asset_b = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "second.png")
    _write_catalog(
        repo_root,
        [
            _case_payload(
                "available_safe_case",
                _asset_payload(
                    "asset_a",
                    asset_a.relative_to(repo_root).as_posix(),
                    source_path="D:/moved-raw/probe.png",
                ),
                _asset_payload(
                    "asset_b",
                    asset_b.relative_to(repo_root).as_posix(),
                    capture="roll",
                    source_path="D:/moved-raw/reference.png",
                ),
            ),
            _case_payload(
                "not_available_case",
                _asset_payload("asset_c", asset_a.relative_to(repo_root).as_posix()),
                _asset_payload("asset_d", asset_b.relative_to(repo_root).as_posix()),
                availability_status="missing",
            ),
            _case_payload(
                "not_safe_case",
                _asset_payload("asset_e", asset_a.relative_to(repo_root).as_posix()),
                _asset_payload("asset_f", asset_b.relative_to(repo_root).as_posix()),
                is_demo_safe=False,
            ),
        ],
    )
    _configure_demo_store(monkeypatch, repo_root)

    response = demo_cases()

    assert [item.id for item in response.cases] == ["available_safe_case"]
    assert response.cases[0].benchmark_score == pytest.approx(0.991)
    file_response = demo_case_asset("available_safe_case", "a")
    assert isinstance(file_response, FileResponse)
    assert Path(file_response.path).resolve() == asset_a.resolve()


def test_load_demo_cases_canonicalizes_raw_benchmark_method_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    asset_a = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "alias_a.png")
    asset_b = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "alias_b.png")
    _write_catalog(
        repo_root,
        [
            _case_payload(
                "alias_case",
                _asset_payload("asset_a", asset_a.relative_to(repo_root).as_posix()),
                _asset_payload("asset_b", asset_b.relative_to(repo_root).as_posix(), capture="roll"),
                recommended_method="dl_quick",
                benchmark_context={
                    "benchmark_run": "full_synthetic_demo_h1",
                    "benchmark_method": "dl_quick",
                    "benchmark_score": 0.991,
                },
            )
        ],
    )
    _configure_demo_store(monkeypatch, repo_root)

    response = demo_store.load_demo_cases()

    assert [item.id for item in response.cases] == ["alias_case"]
    assert response.cases[0].recommended_method == demo_store.MatchMethod.dl
    assert response.cases[0].benchmark_method == "dl_quick"


def test_load_demo_cases_maps_classic_v2_provenance_to_classic_gftt_orb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    asset_a = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "classic_alias_a.png")
    asset_b = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "classic_alias_b.png")
    _write_catalog(
        repo_root,
        [
            _case_payload(
                "classic_alias_case",
                _asset_payload("asset_a", asset_a.relative_to(repo_root).as_posix()),
                _asset_payload("asset_b", asset_b.relative_to(repo_root).as_posix(), capture="roll"),
                recommended_method="classic_v2",
                benchmark_context={
                    "benchmark_run": "full_synthetic_demo_h1",
                    "benchmark_method": "classic_v2",
                    "benchmark_score": 0.812,
                },
            )
        ],
    )
    _configure_demo_store(monkeypatch, repo_root)

    response = demo_store.load_demo_cases()

    assert [item.id for item in response.cases] == ["classic_alias_case"]
    assert response.cases[0].recommended_method == demo_store.MatchMethod.classic_gftt_orb
    assert response.cases[0].benchmark_method == "classic_v2"


def test_load_demo_cases_exposes_public_evidence_quality_and_catalog_build_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    asset_a = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "strong_a.png")
    asset_b = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "strong_b.png")
    asset_c = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "fallback_a.png")
    asset_d = _write_binary_asset(repo_root / "data" / "samples" / "assets" / "synthetic_demo" / "fallback_b.png")
    _write_catalog(
        repo_root,
        [
            _case_payload(
                "strong_case",
                _asset_payload("strong_asset_a", asset_a.relative_to(repo_root).as_posix()),
                _asset_payload("strong_asset_b", asset_b.relative_to(repo_root).as_posix(), capture="roll"),
                selection_diagnostics={
                    "selection_driver": "benchmark_driven",
                    "benchmark_discovery_outcome": "benchmark_best_resolved",
                    "benchmark_selection_status": "benchmark_score_used",
                    "benchmark_backed_selection": True,
                    "heuristic_fallback_used": False,
                },
            ),
            _case_payload(
                "fallback_case",
                _asset_payload("fallback_asset_a", asset_c.relative_to(repo_root).as_posix()),
                _asset_payload("fallback_asset_b", asset_d.relative_to(repo_root).as_posix(), capture="roll"),
                selection_diagnostics={
                    "selection_driver": "heuristic_fallback",
                    "benchmark_discovery_outcome": "dataset_fallback_no_benchmark_evidence",
                    "benchmark_selection_status": "benchmark_resolution_missing",
                    "benchmark_backed_selection": False,
                    "heuristic_fallback_used": True,
                    "fallback_category": "benchmark_resolution_missing",
                },
            ),
        ],
        source_datasets=[
            {
                "dataset": "synthetic_demo",
                "dataset_label": "Synthetic Demo",
                "verify_selection_diagnostics": [
                    {
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
            }
        ],
        metadata={
            "catalog_build_health": {
                "status": "degraded",
                "total_verify_cases_planned": 2,
                "total_verify_cases_built": 2,
                "case_selection_driver_counts": {
                    "benchmark_driven": 1,
                    "heuristic_fallback": 1,
                },
                "datasets_with_missing_benchmark_evidence": ["synthetic_demo:val"],
            }
        },
    )
    _configure_demo_store(monkeypatch, repo_root)

    response = demo_store.load_demo_cases()

    strong_case = next(item for item in response.cases if item.id == "strong_case")
    fallback_case = next(item for item in response.cases if item.id == "fallback_case")

    assert strong_case.evidence_quality is not None
    assert strong_case.evidence_quality.selection_driver == "benchmark_driven"
    assert strong_case.evidence_quality.evidence_status == "strong"
    assert strong_case.evidence_quality.evidence_note == "Selected directly from benchmark evidence."

    assert fallback_case.evidence_quality is not None
    assert fallback_case.evidence_quality.selection_driver == "heuristic_fallback"
    assert fallback_case.evidence_quality.heuristic_fallback_used is True
    assert fallback_case.evidence_quality.evidence_status == "degraded"
    assert "heuristic fallback" in fallback_case.evidence_quality.evidence_note.lower()

    assert response.catalog_build_health is not None
    assert response.catalog_build_health.catalog_build_status == "degraded"
    assert response.catalog_build_health.total_verify_cases == 2
    assert response.catalog_build_health.benchmark_backed_case_count == 1
    assert response.catalog_build_health.heuristic_fallback_case_count == 1
    assert response.catalog_build_health.datasets_with_missing_benchmark_evidence == ["synthetic_demo"]
