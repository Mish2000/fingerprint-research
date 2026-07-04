from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines.ingest.prepare_data_polyu_cross import (
    MANIFEST_COLUMNS,
    PAIR_COLUMNS,
    assign_split,
    build_manifest,
    build_split_pairs,
    build_stats,
    finalize_pair_bundle,
    make_negative_pairs,
    make_positive_pairs,
    sanity_checks,
    split_by_subject,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _write_subject(root: Path, subject_id: int) -> None:
    contactless_dir = root / "contactless_2d_fingerprint_images"
    contact_based_dir = root / "contact-based_fingerprints"
    for session in ("first_session", "second_session"):
        for sample_id in (1, 2):
            _touch(contactless_dir / session / f"p{subject_id}" / f"p{sample_id}.bmp")
            _touch(contact_based_dir / session / f"{subject_id}_{sample_id}.jpg")


def _manifest_with_manual_splits(tmp_path: Path) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    raw_root = tmp_path / "raw"
    for subject_id in range(1, 7):
        _write_subject(raw_root, subject_id)
    manifest = build_manifest(
        raw_root / "contactless_2d_fingerprint_images",
        raw_root / "contact-based_fingerprints",
        path_base=raw_root,
    )
    split_map = {"train": [1, 2], "val": [3, 4], "test": [5, 6]}
    return assign_split(manifest, split_map), split_map


def test_prepare_data_polyu_cross_manifest_infers_sessions_and_sample_uid(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    contactless_dir = raw_root / "contactless_2d_fingerprint_images"
    contact_based_dir = raw_root / "contact-based_fingerprints"

    exemplar_paths = {
        _touch(contactless_dir / "First_Session" / "p1" / "p1.bmp"): "session_1",
        _touch(contactless_dir / "second-session" / "p1" / "p1.bmp"): "session_2",
        _touch(contact_based_dir / "1st_session" / "1_1.jpg"): "session_1",
        _touch(contact_based_dir / "2ND_SESSION" / "1_1.jpg"): "session_2",
    }

    for subject_id in (2, 3):
        _touch(contactless_dir / "First_Session" / f"p{subject_id}" / "p1.bmp")
        _touch(contactless_dir / "second-session" / f"p{subject_id}" / "p1.bmp")
        _touch(contact_based_dir / "1st_session" / f"{subject_id}_1.jpg")
        _touch(contact_based_dir / "2ND_SESSION" / f"{subject_id}_1.jpg")

    manifest = build_manifest(contactless_dir, contact_based_dir, path_base=raw_root)
    split_map = split_by_subject(manifest, seed=42, train_ratio=0.80, val_ratio=0.10)
    manifest = assign_split(manifest, split_map)

    assert manifest.columns.tolist() == MANIFEST_COLUMNS
    assert set(manifest["session_id"].unique()) == {"session_1", "session_2"}
    assert set(manifest["session"].unique()) == {1, 2}
    assert manifest["sample_uid"].is_unique
    assert set(manifest["frgp"].unique()) == {0}
    assert (manifest["finger_unit_id"] == manifest["subject_id"]).all()

    manifest_by_path = manifest.set_index("path")
    for path, expected_session in exemplar_paths.items():
        rel_path = path.relative_to(raw_root).as_posix()
        assert manifest_by_path.loc[rel_path, "session_id"] == expected_session

    subject_one_rows = manifest[(manifest["subject_id"] == 1) & (manifest["sample_id"] == 1)]
    assert set(subject_one_rows["session_id"].tolist()) == {"session_1", "session_2"}


def test_prepare_data_polyu_cross_pair_bundle_phase2a_invariants(tmp_path: Path) -> None:
    manifest, split_map = _manifest_with_manual_splits(tmp_path)
    pos_raw = make_positive_pairs(manifest, max_pos_per_subject=0)
    neg_raw = make_negative_pairs(manifest, pos_raw, seed=13, neg_per_pos=2)
    pos, neg, all_pairs = finalize_pair_bundle(pos_raw, neg_raw)

    assert all_pairs.columns.tolist() == PAIR_COLUMNS
    assert manifest["sample_uid"].is_unique
    assert all_pairs["pair_id"].is_unique
    assert all_pairs.duplicated(subset=["label", "split", "sample_uid_a", "sample_uid_b", "path_a", "path_b"]).sum() == 0

    split_subjects = {split: set(subjects) for split, subjects in split_map.items()}
    assert not (split_subjects["train"] & split_subjects["val"])
    assert not (split_subjects["train"] & split_subjects["test"])
    assert not (split_subjects["val"] & split_subjects["test"])

    assert not pos.empty
    assert not neg.empty
    assert (pos["subject_a"] == pos["subject_b"]).all()
    assert (pos["finger_unit_a"] == pos["finger_unit_b"]).all()
    assert (neg["subject_a"] != neg["subject_b"]).all()
    assert (neg["finger_unit_a"] != neg["finger_unit_b"]).all()

    assert set(all_pairs["modality_a"]) == {"contactless_2d"}
    assert set(all_pairs["modality_b"]) == {"contact_based_2d"}
    assert set(all_pairs["session_a"]) == {"session_1", "session_2"}
    assert set(all_pairs["session_b"]) == {"session_1", "session_2"}

    split_pair_ids: list[int] = []
    for split in ("train", "val", "test"):
        split_pairs = build_split_pairs(pos, neg, split)
        assert not split_pairs.empty
        assert split_pairs.columns.tolist() == PAIR_COLUMNS
        assert set(split_pairs["split"]) == {split}
        split_pair_ids.extend(split_pairs["pair_id"].astype(int).tolist())
    assert len(split_pair_ids) == len(set(split_pair_ids))
    assert set(split_pair_ids) == set(all_pairs["pair_id"].astype(int).tolist())

    sanity = sanity_checks(manifest, split_map, pos, neg, all_pairs)
    assert sanity["ok"] is True
    assert sanity["pair_id_globally_unique"] is True
    assert sanity["duplicate_pair_rows"] == 0
    assert sanity["bad_modality_direction_rows"] == 0
    assert sanity["negative_same_finger_unit"] == 0

    stats = build_stats(manifest, pos, neg, max_pos_per_subject=0, neg_per_pos=2)
    assert stats["contactless_probes_available"] == 24
    assert stats["contactless_probes_used"] == 24
    assert stats["deliberate_positive_subset"] is False
    assert "without replacement" in stats["negative_sampling_policy"]


def test_prepare_data_polyu_cross_positive_cap_is_documented_subset(tmp_path: Path) -> None:
    manifest, _ = _manifest_with_manual_splits(tmp_path)
    pos_raw = make_positive_pairs(manifest, max_pos_per_subject=1)
    neg_raw = make_negative_pairs(manifest, pos_raw, seed=13, neg_per_pos=1)
    pos, neg, _ = finalize_pair_bundle(pos_raw, neg_raw)

    stats = build_stats(manifest, pos, neg, max_pos_per_subject=1, neg_per_pos=1)
    assert stats["deliberate_positive_subset"] is True
    assert stats["contactless_probes_used"] == 6
    assert stats["max_pos_per_subject"] == 1
