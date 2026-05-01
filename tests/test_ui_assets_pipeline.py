from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import io as scipy_io

from src.fpbench.ui_assets.pipeline import UiAssetConfig, build_dataset_ui_assets, build_ui_assets


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_png(path: Path, seed: int, *, color: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    if color:
        image = rng.integers(0, 255, size=(96, 144, 3), dtype=np.uint8)
    else:
        image = rng.integers(0, 255, size=(80, 120), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    assert ok, f"failed to write synthetic image at {path}"
    return path


def _write_surface_mat(path: Path, surface: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy_io.savemat(path, {"surface": surface})
    return path


def _make_repo_layout(repo_root: Path, dataset: str) -> None:
    (repo_root / "data" / "manifests" / dataset).mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "processed" / dataset).mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "samples" / "assets").mkdir(parents=True, exist_ok=True)
    (repo_root / "artifacts" / "reports" / "benchmark").mkdir(parents=True, exist_ok=True)


def _write_manifest(repo_root: Path, dataset: str, rows: list[dict]) -> None:
    manifest_path = repo_root / "data" / "manifests" / dataset / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "capture",
                "subject_id",
                "impression",
                "ppi",
                "frgp",
                "path",
                "split",
                "sample_id",
                "session",
                "source_modality",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _base_row(dataset: str, path: Path, *, subject_id: int, split: str, capture: str, frgp: int = 1, modality: str = "optical_2d") -> dict:
    return {
        "dataset": dataset,
        "capture": capture,
        "subject_id": str(subject_id),
        "impression": "sample_01",
        "ppi": "500",
        "frgp": str(frgp),
        "path": str(path),
        "split": split,
        "sample_id": str(subject_id),
        "session": "1",
        "source_modality": modality,
    }


def test_ui_assets_pipeline_builds_index_and_assets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dataset = "mini_dataset"
    _make_repo_layout(repo_root, dataset)

    img_a = _write_png(repo_root / "data" / "raw" / dataset / "train_a.png", 1)
    img_b = _write_png(repo_root / "data" / "raw" / dataset / "val_b.png", 2, color=True)
    _write_manifest(
        repo_root,
        dataset,
        [
            _base_row(dataset, img_a, subject_id=101, split="train", capture="plain", frgp=1),
            _base_row(dataset, img_b, subject_id=202, split="val", capture="contactless", frgp=2, modality="contactless"),
        ],
    )
    _write_json(repo_root / "data" / "manifests" / dataset / "split.json", {"train": [101], "val": [202], "test": []})
    _write_json(repo_root / "data" / "manifests" / dataset / "stats.json", {"manifest_rows": 2, "unique_subjects": 2})

    result = build_dataset_ui_assets(dataset, repo_root=repo_root, config=UiAssetConfig(max_items_per_dataset=10))

    assert result["validation_status"] == "pass"
    index_path = repo_root / result["index_path"]
    report_path = repo_root / result["validation_report_path"]
    assert index_path.exists()
    assert report_path.exists()

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["dataset"] == dataset
    assert index_payload["selection_policy"].startswith("deterministic_round_robin")
    assert index_payload["validation_status"] == "pass"
    assert len(index_payload["items"]) == 2

    for item in index_payload["items"]:
        assert item["asset_id"]
        assert item["source_path"].startswith("data/raw/")
        assert (repo_root / item["thumbnail_path"]).exists()
        assert (repo_root / item["preview_path"]).exists()
        thumb = cv2.imread(str(repo_root / item["thumbnail_path"]), cv2.IMREAD_UNCHANGED)
        preview = cv2.imread(str(repo_root / item["preview_path"]), cv2.IMREAD_UNCHANGED)
        assert thumb.shape[0] == thumb.shape[1] == 160
        assert preview.shape[0] == preview.shape[1] == 512


def test_ui_assets_pipeline_handles_missing_and_corrupt_sources_without_global_failure(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dataset = "faulty_dataset"
    _make_repo_layout(repo_root, dataset)

    good_image = _write_png(repo_root / "data" / "raw" / dataset / "good.png", 3)
    corrupt_image = repo_root / "data" / "raw" / dataset / "corrupt.png"
    corrupt_image.parent.mkdir(parents=True, exist_ok=True)
    corrupt_image.write_text("not-an-image", encoding="utf-8")
    missing_image = repo_root / "data" / "raw" / dataset / "missing.png"
    rows = [
        _base_row(dataset, good_image, subject_id=1, split="train", capture="plain"),
        _base_row(dataset, corrupt_image, subject_id=2, split="val", capture="plain"),
        _base_row(dataset, missing_image, subject_id=3, split="test", capture="roll"),
        {
            **_base_row(dataset, good_image, subject_id=4, split="train", capture="roll"),
            "path": "",
        },
    ]
    _write_manifest(repo_root, dataset, rows)
    _write_json(repo_root / "data" / "manifests" / dataset / "split.json", {"train": [1, 4], "val": [2], "test": [3]})
    _write_json(repo_root / "data" / "manifests" / dataset / "stats.json", {"manifest_rows": 4, "unique_subjects": 4})

    result = build_dataset_ui_assets(dataset, repo_root=repo_root, config=UiAssetConfig(max_items_per_dataset=10))

    index_payload = json.loads((repo_root / result["index_path"]).read_text(encoding="utf-8"))
    report_payload = json.loads((repo_root / result["validation_report_path"]).read_text(encoding="utf-8"))
    assert len(index_payload["items"]) == 1
    assert report_payload["generated_items"] == 1
    assert report_payload["missing_source_files"] == 1
    assert report_payload["unreadable_source_files"] == 1
    assert report_payload["missing_critical_metadata"] == 1
    assert report_payload["validation_status"] == "pass_with_warnings"


def test_ui_assets_pipeline_rerun_is_logically_deterministic_and_samples_are_unchanged(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dataset = "deterministic_dataset"
    _make_repo_layout(repo_root, dataset)

    sample_catalog = repo_root / "data" / "samples" / "catalog.json"
    sample_catalog.parent.mkdir(parents=True, exist_ok=True)
    sample_catalog.write_text('{"catalog":"unchanged"}\n', encoding="utf-8")
    sample_asset = repo_root / "data" / "samples" / "assets" / "demo_asset.txt"
    sample_asset.write_text("keep me", encoding="utf-8")

    images = [
        _write_png(repo_root / "data" / "raw" / dataset / "train_subject_1.png", 10),
        _write_png(repo_root / "data" / "raw" / dataset / "val_subject_2.png", 11),
        _write_png(repo_root / "data" / "raw" / dataset / "test_subject_3.png", 12),
    ]
    _write_manifest(
        repo_root,
        dataset,
        [
            _base_row(dataset, images[0], subject_id=1, split="train", capture="plain"),
            _base_row(dataset, images[1], subject_id=2, split="val", capture="contactless", modality="contactless"),
            _base_row(dataset, images[2], subject_id=3, split="test", capture="roll"),
        ],
    )
    _write_json(repo_root / "data" / "manifests" / dataset / "split.json", {"train": [1], "val": [2], "test": [3]})
    _write_json(repo_root / "data" / "manifests" / dataset / "stats.json", {"manifest_rows": 3, "unique_subjects": 3})

    first = build_dataset_ui_assets(dataset, repo_root=repo_root, config=UiAssetConfig(max_items_per_dataset=2))
    second = build_dataset_ui_assets(dataset, repo_root=repo_root, config=UiAssetConfig(max_items_per_dataset=2))

    first_index = json.loads((repo_root / first["index_path"]).read_text(encoding="utf-8"))
    second_index = json.loads((repo_root / second["index_path"]).read_text(encoding="utf-8"))
    assert [item["asset_id"] for item in first_index["items"]] == [item["asset_id"] for item in second_index["items"]]
    assert [item["thumbnail_path"] for item in first_index["items"]] == [item["thumbnail_path"] for item in second_index["items"]]
    assert [item["preview_path"] for item in first_index["items"]] == [item["preview_path"] for item in second_index["items"]]
    assert sample_catalog.read_text(encoding="utf-8") == '{"catalog":"unchanged"}\n'
    assert sample_asset.read_text(encoding="utf-8") == "keep me"


def test_ui_assets_registry_is_derived_and_does_not_depend_on_benchmark_or_demo_layers(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dataset = "registry_dataset"
    _make_repo_layout(repo_root, dataset)

    image = _write_png(repo_root / "data" / "raw" / dataset / "single.png", 20)
    _write_manifest(repo_root, dataset, [_base_row(dataset, image, subject_id=11, split="train", capture="plain")])
    _write_json(repo_root / "data" / "manifests" / dataset / "split.json", {"train": [11], "val": [], "test": []})
    _write_json(repo_root / "data" / "manifests" / dataset / "stats.json", {"manifest_rows": 1, "unique_subjects": 1})

    benchmark_poison = repo_root / "artifacts" / "reports" / "benchmark" / "scores_poison.csv"
    benchmark_poison.parent.mkdir(parents=True, exist_ok=True)
    benchmark_poison.write_text("label,score,path_a,path_b\n1,999,a,b\n", encoding="utf-8")
    demo_catalog = repo_root / "data" / "samples" / "catalog.json"
    demo_catalog.write_text('{"demo":"should stay unused"}\n', encoding="utf-8")

    registry = build_ui_assets([dataset], repo_root=repo_root, config=UiAssetConfig(max_items_per_dataset=5))

    registry_path = repo_root / "data" / "processed" / "ui_assets_registry.json"
    assert registry_path.exists()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["datasets"][0]["dataset"] == dataset
    assert payload["datasets"][0]["index_path"].endswith("/ui_assets/index.json")
    assert benchmark_poison.read_text(encoding="utf-8") == "label,score,path_a,path_b\n1,999,a,b\n"
    assert demo_catalog.read_text(encoding="utf-8") == '{"demo":"should stay unused"}\n'
    assert registry["datasets"][0]["item_count"] == 1


def test_ui_assets_pipeline_polyu_3d_renders_surface_previews_and_updates_registry(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dataset = "polyu_3d"
    _make_repo_layout(repo_root, dataset)

    raw_surface = _write_surface_mat(
        repo_root / "data" / "raw" / dataset / "subject_001_surface.mat",
        np.array(
            [
                [0.0, 1.0, 2.0, np.nan],
                [0.5, 1.5, 2.5, 3.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    raw_photometric = _write_png(repo_root / "data" / "raw" / dataset / "subject_001_raw.bmp", 31)

    rows = [
        {
            **_base_row(
                dataset,
                raw_surface,
                subject_id=1,
                split="val",
                capture="contactless",
                frgp=0,
                modality="contactless_3d_surface",
            ),
            "ppi": "0",
            "session": "1",
        },
        {
            **_base_row(
                dataset,
                raw_photometric,
                subject_id=1,
                split="val",
                capture="contactless",
                frgp=0,
                modality="contactless_2d_photometric_raw",
            ),
            "ppi": "0",
            "session": "1",
        },
    ]
    _write_manifest(repo_root, dataset, rows)
    _write_json(repo_root / "data" / "manifests" / dataset / "split.json", {"train": [], "val": [1], "test": []})
    _write_json(repo_root / "data" / "manifests" / dataset / "stats.json", {"manifest_rows": 2, "unique_subjects": 1})

    registry = build_ui_assets([dataset], repo_root=repo_root, config=UiAssetConfig(max_items_per_dataset=10))
    result = registry["datasets"][0]
    index_payload = json.loads((repo_root / result["index_path"]).read_text(encoding="utf-8"))
    report_payload = json.loads((repo_root / result["validation_report_path"]).read_text(encoding="utf-8"))

    assert report_payload["canonical_ui_policy"]["name"] == "canonical_surface_only"
    assert result["validation_status"] == "pass"
    assert result["item_count"] == 1
    assert result["browser_preview_enabled"] is True
    assert index_payload["browser_preview_enabled"] is True
    assert index_payload["summary"]["items_by_render_strategy"] == {"deterministic_numeric_artifact": 1}
    assert ".mat" in index_payload["deterministic_render_source_suffixes"]
    assert report_payload["canonical_render_attempts"] == 1
    assert report_payload["canonical_render_failures"] == 0
    assert report_payload["rendered_items_by_strategy"] == {"deterministic_numeric_artifact": 1}
    assert all(item["modality"] == "contactless_3d_surface" for item in index_payload["items"])
    assert all(item["render_strategy"] == "deterministic_numeric_artifact" for item in index_payload["items"])
    assert all("contactless_2d_photometric_raw" not in item["modality"] for item in index_payload["items"])

    item = index_payload["items"][0]
    assert (repo_root / item["thumbnail_path"]).is_file()
    assert (repo_root / item["preview_path"]).is_file()
    thumb = cv2.imread(str(repo_root / item["thumbnail_path"]), cv2.IMREAD_UNCHANGED)
    preview = cv2.imread(str(repo_root / item["preview_path"]), cv2.IMREAD_UNCHANGED)
    assert thumb is not None
    assert preview is not None
    assert thumb.shape[:2] == (160, 160)
    assert preview.shape[:2] == (512, 512)


def test_ui_assets_pipeline_polyu_3d_stays_fail_closed_after_surface_render_attempt(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dataset = "polyu_3d"
    _make_repo_layout(repo_root, dataset)

    raw_surface = repo_root / "data" / "raw" / dataset / "subject_001_surface.mat"
    raw_surface.parent.mkdir(parents=True, exist_ok=True)
    raw_surface.write_text("not-a-real-mat", encoding="utf-8")
    raw_photometric = _write_png(repo_root / "data" / "raw" / dataset / "subject_001_raw.bmp", 32)

    rows = [
        {
            **_base_row(
                dataset,
                raw_surface,
                subject_id=1,
                split="val",
                capture="contactless",
                frgp=0,
                modality="contactless_3d_surface",
            ),
            "ppi": "0",
            "session": "1",
        },
        {
            **_base_row(
                dataset,
                raw_photometric,
                subject_id=1,
                split="val",
                capture="contactless",
                frgp=0,
                modality="contactless_2d_photometric_raw",
            ),
            "ppi": "0",
            "session": "1",
        },
    ]
    _write_manifest(repo_root, dataset, rows)
    _write_json(repo_root / "data" / "manifests" / dataset / "split.json", {"train": [], "val": [1], "test": []})
    _write_json(repo_root / "data" / "manifests" / dataset / "stats.json", {"manifest_rows": 2, "unique_subjects": 1})

    result = build_dataset_ui_assets(dataset, repo_root=repo_root, config=UiAssetConfig(max_items_per_dataset=10))
    index_payload = json.loads((repo_root / result["index_path"]).read_text(encoding="utf-8"))
    report_payload = json.loads((repo_root / result["validation_report_path"]).read_text(encoding="utf-8"))

    assert report_payload["canonical_ui_policy"]["name"] == "canonical_surface_only"
    assert report_payload["selected_records"] == 1
    assert report_payload["canonical_render_attempts"] == 1
    assert report_payload["canonical_render_failures"] == 1
    assert report_payload["unreadable_source_files"] == 1
    assert result["validation_status"] == "fail"
    assert index_payload["items"] == []
    assert index_payload["browser_preview_enabled"] is False
    assert "after 1 deterministic render attempt" in report_payload["browser_preview_exclusion_reason"]
    assert all("contactless_2d_photometric_raw" not in item.get("modality", "") for item in index_payload["items"])


def test_ui_assets_pipeline_unsw_excludes_auxiliary_modalities(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dataset = "unsw_2d3d"
    _make_repo_layout(repo_root, dataset)

    optical = _write_png(repo_root / "data" / "raw" / dataset / "optical.bmp", 41)
    reconstructed = _write_png(repo_root / "data" / "raw" / dataset / "reconstructed.bmp", 42)
    derived = _write_png(repo_root / "data" / "raw" / dataset / "derived.bmp", 43)
    raw_intermediate = _write_png(repo_root / "data" / "raw" / dataset / "raw.bmp", 44)

    rows = [
        _base_row(dataset, optical, subject_id=1, split="val", capture="contact_based", frgp=1, modality="optical_2d"),
        _base_row(dataset, reconstructed, subject_id=1, split="val", capture="contactless", frgp=1, modality="reconstructed_3d"),
        _base_row(dataset, derived, subject_id=1, split="val", capture="contactless", frgp=1, modality="derived_3d_variant"),
        _base_row(dataset, raw_intermediate, subject_id=1, split="val", capture="contactless", frgp=1, modality="reconstruction_intermediate"),
    ]
    _write_manifest(repo_root, dataset, rows)
    _write_json(repo_root / "data" / "manifests" / dataset / "split.json", {"train": [], "val": [1], "test": []})
    _write_json(repo_root / "data" / "manifests" / dataset / "stats.json", {"manifest_rows": 4, "unique_subjects": 1})

    result = build_dataset_ui_assets(dataset, repo_root=repo_root, config=UiAssetConfig(max_items_per_dataset=10))
    index_payload = json.loads((repo_root / result["index_path"]).read_text(encoding="utf-8"))
    report_payload = json.loads((repo_root / result["validation_report_path"]).read_text(encoding="utf-8"))

    assert result["validation_status"] == "pass"
    assert report_payload["canonical_ui_policy"]["name"] == "canonical_optical2d_reconstructed3d_only"
    modalities = {item["modality"] for item in index_payload["items"]}
    assert modalities <= {"optical_2d", "reconstructed_3d"}
    assert "derived_3d_variant" not in modalities
    assert "reconstruction_intermediate" not in modalities
