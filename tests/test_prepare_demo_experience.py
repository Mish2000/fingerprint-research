from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.responses import FileResponse

import apps.api.catalog_store as catalog_store
import apps.api.demo_store as demo_store
from src.fpbench.demo import preparation as prep


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_file(path: Path, payload: bytes = b"asset") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _make_result(
    name: str,
    label: str,
    *,
    warnings: list[str] | None = None,
    failures: list[str] | None = None,
    metrics: dict | None = None,
) -> prep.StepResult:
    result = prep.StepResult(name=name, label=label)
    for message in warnings or []:
        result.add_warning(message)
    for message in failures or []:
        result.add_failure(message)
    if metrics:
        result.metrics.update(metrics)
    return result.finalize()


def _configure_catalog_roots(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    for module in (demo_store, catalog_store):
        monkeypatch.setattr(module, "ROOT", repo_root)
        monkeypatch.setattr(module, "SAMPLES_ROOT", repo_root / "data" / "samples")
        monkeypatch.setattr(module, "CATALOG_PATH", repo_root / "data" / "samples" / "catalog.json")
        monkeypatch.setattr(module, "ASSETS_ROOT", repo_root / "data" / "samples" / "assets")
    monkeypatch.setattr(catalog_store, "PROCESSED_ROOT", repo_root / "data" / "processed")
    monkeypatch.setattr(catalog_store, "UI_ASSETS_REGISTRY_PATH", repo_root / "data" / "processed" / "ui_assets_registry.json")
    catalog_store.clear_catalog_store_cache()
    demo_store.clear_demo_store_cache()


def test_prerequisites_fail_when_manifest_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    for path in (
        repo_root / "data" / "manifests" / "demo_ds",
        repo_root / "data" / "samples",
        repo_root / "data" / "processed",
        repo_root / "scripts",
        repo_root / "apps" / "api",
        repo_root / "src",
    ):
        path.mkdir(parents=True, exist_ok=True)

    for path in (
        repo_root / "scripts" / "build_demo_catalog.py",
        repo_root / "scripts" / "build_ui_assets.py",
        repo_root / "apps" / "api" / "main.py",
        repo_root / "apps" / "api" / "identify_demo_store.py",
        repo_root / "apps" / "api" / "catalog_store.py",
        repo_root / "data" / "manifests" / "demo_ds" / "stats.json",
        repo_root / "data" / "manifests" / "demo_ds" / "split.json",
        repo_root / "data" / "manifests" / "demo_ds" / "pairs_val.csv",
    ):
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(prep, "IDENTITY_DATASET_ORDER", ["demo_ds"])
    monkeypatch.setattr(prep, "VERIFY_CASE_PLANS", [{"dataset": "demo_ds", "split": "val", "case_type": "hero"}])

    ctx = prep.PrepareContext(
        options=prep.PrepareOptions(
            repo_root=repo_root,
            report_root=repo_root / "artifacts" / "reports" / "demo",
            emit=lambda *_: None,
        ),
        emit=lambda *_: None,
    )
    result = prep._step_prerequisites(ctx)

    assert result.status == "failed"
    assert any("manifest.csv" in message for message in result.failures)


def test_runner_stops_after_blocking_failure_and_marks_later_steps_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []

    def pass_step(ctx: prep.PrepareContext) -> prep.StepResult:
        call_order.append("pass")
        return _make_result("one", "One", metrics={"datasets_checked": 1})

    def fail_step(ctx: prep.PrepareContext) -> prep.StepResult:
        call_order.append("fail")
        return _make_result("two", "Two", failures=["blocking failure"])

    def after_step(ctx: prep.PrepareContext) -> prep.StepResult:
        call_order.append("after")
        return _make_result("three", "Three")

    monkeypatch.setattr(
        prep,
        "_selected_steps",
        lambda mode: [("one", "One", pass_step), ("two", "Two", fail_step), ("three", "Three", after_step)],
    )

    summary = prep.run_prepare_demo_experience(
        prep.PrepareOptions(
            mode=prep.MODE_FULL,
            report_root=tmp_path / "reports",
            emit=lambda *_: None,
        )
    )

    assert call_order == ["pass", "fail"]
    assert [step["status"] for step in summary["steps"]] == ["passed", "failed", "skipped"]
    assert summary["verdict"] == "not ready"
    assert Path(summary["report_path"]).is_file()


def test_rebuild_only_success_does_not_claim_demo_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prep,
        "_selected_steps",
        lambda mode: [
            ("prerequisites", "Prerequisites", lambda ctx: _make_result("prerequisites", "Prerequisites")),
            ("catalog", "Catalog", lambda ctx: _make_result("catalog", "Catalog")),
        ],
    )

    summary = prep.run_prepare_demo_experience(
        prep.PrepareOptions(
            mode=prep.MODE_REBUILD_ONLY,
            report_root=tmp_path / "reports",
            emit=lambda *_: None,
        )
    )

    assert summary["status"] == "success"
    assert summary["verdict"] == "not ready"
    assert "does not run the full demo-readiness evaluation" in summary["verdict_reason"]


def test_full_warning_only_run_is_demo_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prep,
        "_selected_steps",
        lambda mode: [
            ("prerequisites", "Prerequisites", lambda ctx: _make_result("prerequisites", "Prerequisites")),
            ("raw_data", "Raw Data Verification", lambda ctx: _make_result("raw_data", "Raw Data Verification")),
            ("catalog", "Catalog", lambda ctx: _make_result("catalog", "Catalog")),
            ("ui_assets", "UI Assets", lambda ctx: _make_result("ui_assets", "UI Assets", metrics={"assets_built": 12})),
            ("sanity", "Sanity", lambda ctx: _make_result("sanity", "Sanity", metrics={"verify_cases_checked": 4})),
            (
                "seed",
                "Seed",
                lambda ctx: _make_result(
                    "seed",
                    "Seed",
                    warnings=["non-demo identities remain"],
                    metrics={"demo_seeded_count": 2},
                ),
            ),
            (
                "api_health",
                "API Health",
                lambda ctx: _make_result("api_health", "API Health", metrics={"demo_seeded_count": 2}),
            ),
        ],
    )

    summary = prep.run_prepare_demo_experience(
        prep.PrepareOptions(
            mode=prep.MODE_FULL,
            report_root=tmp_path / "reports",
            emit=lambda *_: None,
        )
    )

    assert summary["status"] == "partial"
    assert summary["verdict"] == "demo ready"
    assert summary["totals"]["warnings"] == 1


def test_sanity_step_flags_missing_demo_asset_and_unseedable_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    _configure_catalog_roots(monkeypatch, repo_root)

    existing_asset = _write_file(repo_root / "data" / "samples" / "assets" / "demo_ds" / "right.png")
    _write_json(
        repo_root / "data" / "samples" / "catalog.json",
        {
            "catalog_version": "1.0.0",
            "generated_at": "2026-04-02T00:00:00Z",
            "source_datasets": [{"dataset": "demo_ds", "dataset_label": "Demo Dataset"}],
            "verify_cases": [
                {
                    "case_id": "case_demo_safe",
                    "title": "Demo Case",
                    "description": "Broken demo case.",
                    "dataset": "demo_ds",
                    "split": "val",
                    "case_type": "easy_genuine",
                    "difficulty": "easy",
                    "ground_truth": "match",
                    "recommended_method": "sift",
                    "capture_a": "plain",
                    "capture_b": "roll",
                    "image_a": {
                        "asset_id": "asset_missing",
                        "path": "data/samples/assets/demo_ds/missing.png",
                        "relative_path": "data/samples/assets/demo_ds/missing.png",
                        "availability_status": "available",
                    },
                    "image_b": {
                        "asset_id": "asset_right",
                        "path": existing_asset.relative_to(repo_root).as_posix(),
                        "relative_path": existing_asset.relative_to(repo_root).as_posix(),
                        "availability_status": "available",
                    },
                    "is_demo_safe": True,
                    "availability_status": "available",
                    "selection_reason": "Chosen for demo.",
                    "selection_policy": "demo",
                    "tags": ["hero"],
                    "modality_relation": "same_modality",
                }
            ],
            "identify_gallery": {
                "identities": [
                    {
                        "identity_id": "identity_demo_safe",
                        "dataset": "demo_ds",
                        "display_name": "Demo Identity",
                        "subject_id": 101,
                        "gallery_role": "standard",
                        "enrollment_candidates": ["asset_enroll"],
                        "probe_candidates": ["asset_probe"],
                        "recommended_enrollment_asset_id": "asset_enroll",
                        "recommended_probe_asset_id": "asset_probe",
                        "tags": ["demo_ds"],
                        "is_demo_safe": True,
                        "exemplars": [
                            {
                                "asset_id": "asset_enroll",
                                "dataset": "demo_ds",
                                "path": existing_asset.relative_to(repo_root).as_posix(),
                                "relative_path": existing_asset.relative_to(repo_root).as_posix(),
                                "source_path": "data/raw/demo_ds/enroll.png",
                                "source_relative_path": "data/raw/demo_ds/enroll.png",
                                "availability_status": "available",
                                "capture": "plain",
                                "traceability": {"source_dataset": "demo_ds"},
                            }
                        ],
                    }
                ],
                "demo_scenarios": [],
            },
            "dataset_browser_seed": [],
            "metadata": {
                "total_verify_cases": 1,
                "total_identity_records": 1,
                "total_browser_seed_items": 0,
                "included_datasets": ["demo_ds"],
                "excluded_datasets": [],
                "validation_status": "pass",
                "validation_errors_count": 0,
                "validation_warnings_count": 0,
            },
        },
    )

    ctx = prep.PrepareContext(
        options=prep.PrepareOptions(
            repo_root=repo_root,
            report_root=repo_root / "artifacts" / "reports" / "demo",
            emit=lambda *_: None,
        ),
        emit=lambda *_: None,
    )
    result = prep._step_sanity(ctx)

    assert result.status == "failed"
    assert any("missing asset" in message or "No verify demo cases" in message for message in result.failures)
    assert any("No demo-safe identification identities" in message for message in result.failures)


def test_seed_step_warns_when_non_demo_identities_remain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prep, "_ensure_api_started", lambda ctx, step: True)
    monkeypatch.setattr(api_main := prep.api_main, "identify_demo_reset", lambda: SimpleNamespace(removed_count=1))
    monkeypatch.setattr(
        api_main,
        "identify_demo_seed",
        lambda: SimpleNamespace(seeded_count=2, updated_count=0, total_enrolled=3, demo_seeded_count=2),
    )
    monkeypatch.setattr(api_main, "identify_stats", lambda: SimpleNamespace(demo_seeded_count=2))

    ctx = prep.PrepareContext(options=prep.PrepareOptions(emit=lambda *_: None), emit=lambda *_: None)
    result = prep._step_seed(ctx)

    assert result.status == "passed_with_warnings"
    assert result.metrics["demo_seeded_count"] == 2
    assert any("non-demo identities" in warning for warning in result.warnings)


def test_api_health_step_requires_demo_facing_endpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(prep, "_ensure_api_started", lambda ctx, step: True)

    served_asset = _write_file(tmp_path / "thumb.png")
    monkeypatch.setattr(prep.api_main, "health", lambda: {"ok": True, "identify_ok": True, "error": None, "identify_error": None})
    monkeypatch.setattr(
        prep.api_main,
        "catalog_datasets",
        lambda: SimpleNamespace(items=[SimpleNamespace(dataset="demo_ds", has_browser_assets=True)]),
    )
    monkeypatch.setattr(prep.api_main, "catalog_verify_cases", lambda **_: SimpleNamespace(total=1))
    monkeypatch.setattr(prep.api_main, "demo_cases", lambda: SimpleNamespace(cases=[]))
    monkeypatch.setattr(
        prep.api_main,
        "catalog_identify_gallery",
        lambda **_: SimpleNamespace(demo_identities=[], probe_cases=[]),
    )
    monkeypatch.setattr(
        prep.api_main,
        "catalog_dataset_browser",
        lambda **_: SimpleNamespace(total=1, items=[SimpleNamespace(asset_id="asset_1")]),
    )
    monkeypatch.setattr(prep.api_main, "catalog_asset", lambda *_, **__: FileResponse(served_asset))
    monkeypatch.setattr(prep.api_main, "identify_stats", lambda: SimpleNamespace(demo_seeded_count=0))

    ctx = prep.PrepareContext(options=prep.PrepareOptions(emit=lambda *_: None), emit=lambda *_: None)
    result = prep._step_api_health(ctx)

    assert result.status == "failed"
    assert any("/demo/cases returned no demo-ready verify cases" in message for message in result.failures)
    assert any("/catalog/identify-gallery returned no demo identities" in message for message in result.failures)
    assert any("zero demo-seeded identities" in message for message in result.failures)


def test_main_exit_code_follows_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prep, "run_prepare_demo_experience", lambda options=None: {"verdict": "demo ready"})
    assert prep.main([]) == 0

    monkeypatch.setattr(prep, "run_prepare_demo_experience", lambda options=None: {"verdict": "not ready"})
    assert prep.main(["--mode", prep.MODE_REBUILD_ONLY]) == 1
