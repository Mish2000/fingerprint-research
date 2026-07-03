from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.fpbench.datasets.polyu_cross import (
    MANIFEST_COLUMNS,
    PolyUCrossManifestError,
    build_manifest_dataframe,
    write_manifest,
)


def _write_image(path: Path, *, size: tuple[int, int] = (16, 12)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=180).save(path)
    return path


def _fake_polyu_tree(root: Path) -> None:
    _write_image(
        root / "contactless_2d_fingerprint_images" / "first_session" / "p10" / "p2.bmp",
        size=(21, 22),
    )
    _write_image(
        root / "contact-based fingerprints" / "second_session" / "10_2.jpg",
        size=(31, 32),
    )
    _write_image(
        root
        / "processed_contactless_2d_fingerprint_images"
        / "session_2"
        / "subject_11"
        / "finger_3"
        / "capture_4.png",
        size=(41, 42),
    )
    _write_image(
        root / "contact-based_fingerprints" / "session_1" / "subject_11_finger_3_capture_5.jpg",
        size=(51, 52),
    )
    _write_image(root / "contact-based_fingerprints" / "session_1" / "mystery.jpg")


def test_polyu_cross_manifest_discovers_modalities_and_writes_clean_schema(tmp_path: Path) -> None:
    dataset_root = tmp_path / "polyu"
    _fake_polyu_tree(dataset_root)
    output = tmp_path / "out" / "manifest.csv"

    result = write_manifest(dataset_root, output, strict=False)
    manifest = pd.read_csv(output, dtype={"subject_id": str, "finger_id": str, "capture_id": str})

    assert result.row_count == 4
    assert manifest.columns.tolist() == MANIFEST_COLUMNS
    assert set(manifest["modality"].tolist()) == {"contactless_2d", "contact_based_2d"}
    assert set(manifest["source_dataset"].tolist()) == {"polyu_cross"}
    assert manifest["sample_id"].is_unique
    assert not any(Path(value).is_absolute() for value in manifest["image_path"].tolist())

    by_path = manifest.set_index("image_path")
    canonical_contactless = by_path.loc["contactless_2d_fingerprint_images/first_session/p10/p2.bmp"]
    assert canonical_contactless["subject_id"] == "10"
    assert canonical_contactless["finger_id"] == "10"
    assert canonical_contactless["capture_id"] == "2"
    assert canonical_contactless["session_id"] == "session_1"
    assert int(canonical_contactless["width"]) == 21
    assert int(canonical_contactless["height"]) == 22

    canonical_contact_based = by_path.loc["contact-based fingerprints/second_session/10_2.jpg"]
    assert canonical_contact_based["subject_id"] == "10"
    assert canonical_contact_based["finger_id"] == "10"
    assert canonical_contact_based["capture_id"] == "2"
    assert canonical_contact_based["session_id"] == "session_2"

    named_contactless = by_path.loc[
        "processed_contactless_2d_fingerprint_images/session_2/subject_11/finger_3/capture_4.png"
    ]
    assert named_contactless["subject_id"] == "11"
    assert named_contactless["finger_id"] == "3"
    assert named_contactless["capture_id"] == "4"
    assert named_contactless["session_id"] == "session_2"

    named_contact_based = by_path.loc["contact-based_fingerprints/session_1/subject_11_finger_3_capture_5.jpg"]
    assert named_contact_based["subject_id"] == "11"
    assert named_contact_based["finger_id"] == "3"
    assert named_contact_based["capture_id"] == "5"
    assert named_contact_based["session_id"] == "session_1"

    warnings = json.loads(result.warnings_path.read_text(encoding="utf-8"))
    assert len(warnings) == 1
    assert warnings[0]["reason"] == "unparseable_metadata"
    assert warnings[0]["image_path"] == "contact-based_fingerprints/session_1/mystery.jpg"

    sanity = json.loads(result.sanity_report_path.read_text(encoding="utf-8"))
    assert sanity["manifest"]["rows"] == 4
    assert sanity["warnings"]["count"] == 1
    assert sanity["checks"]["columns_exact"] is True
    assert sanity["checks"]["sample_id_unique"] is True
    assert sanity["checks"]["image_paths_relative"] is True


def test_polyu_cross_manifest_order_is_deterministic(tmp_path: Path) -> None:
    dataset_root = tmp_path / "polyu"
    _fake_polyu_tree(dataset_root)

    first, _ = build_manifest_dataframe(dataset_root, strict=False)
    second, _ = build_manifest_dataframe(dataset_root, strict=False)

    assert first.equals(second)
    assert first["image_path"].tolist() == [
        "contact-based fingerprints/second_session/10_2.jpg",
        "contact-based_fingerprints/session_1/subject_11_finger_3_capture_5.jpg",
        "contactless_2d_fingerprint_images/first_session/p10/p2.bmp",
        "processed_contactless_2d_fingerprint_images/session_2/subject_11/finger_3/capture_4.png",
    ]


def test_polyu_cross_manifest_strict_mode_fails_on_unparseable_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "polyu"
    _fake_polyu_tree(dataset_root)

    with pytest.raises(PolyUCrossManifestError, match="unparseable"):
        build_manifest_dataframe(dataset_root, strict=True)


def test_polyu_cross_manifest_script_has_useful_help() -> None:
    script = Path(__file__).resolve().parents[1] / "pipelines" / "manifests" / "build_polyu_cross_manifest.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--dataset-root" in completed.stdout
    assert "--output" in completed.stdout
    assert "--strict" in completed.stdout
    assert "generate pairs" in completed.stdout
