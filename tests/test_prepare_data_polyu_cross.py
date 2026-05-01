from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines.ingest.prepare_data_polyu_cross import assign_split, build_manifest, split_by_subject


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_prepare_data_polyu_cross_manifest_infers_named_sessions(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    contactless_dir = raw_root / "contactless_2d_fingerprint_images"
    contact_based_dir = raw_root / "contact-based_fingerprints"
    out_dir = tmp_path / "out"

    exemplar_paths = {
        _touch(contactless_dir / "First_Session" / "p1" / "p1.bmp").resolve(): 1,
        _touch(contactless_dir / "second-session" / "p1" / "p1.bmp").resolve(): 2,
        _touch(contact_based_dir / "1st_session" / "1_1.jpg").resolve(): 1,
        _touch(contact_based_dir / "2ND_SESSION" / "1_1.jpg").resolve(): 2,
    }

    for subject_id in (2, 3):
        _touch(contactless_dir / "First_Session" / f"p{subject_id}" / "p1.bmp")
        _touch(contactless_dir / "second-session" / f"p{subject_id}" / "p1.bmp")
        _touch(contact_based_dir / "1st_session" / f"{subject_id}_1.jpg")
        _touch(contact_based_dir / "2ND_SESSION" / f"{subject_id}_1.jpg")

    manifest = build_manifest(contactless_dir, contact_based_dir)
    split_map = split_by_subject(manifest, seed=42, train_ratio=0.80, val_ratio=0.10)
    manifest = assign_split(manifest, split_map)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    manifest = pd.read_csv(out_dir / "manifest.csv")
    assert manifest.columns.tolist() == [
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
    ]
    assert set(manifest["session"].unique()) == {1, 2}

    manifest_by_path = manifest.set_index("path")
    for path, expected_session in exemplar_paths.items():
        assert int(manifest_by_path.loc[str(path), "session"]) == expected_session

    subject_one_rows = manifest[(manifest["subject_id"] == 1) & (manifest["sample_id"] == 1)]
    assert set(subject_one_rows["session"].tolist()) == {1, 2}
