from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines.ingest.generate_pairs import _infer_finger_col
from pipelines.ingest.prepare_data_sd300b import (
    build_manifest,
    choose_one,
    make_positive_pairs,
    parse_file,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    return path


def test_parse_file_maps_plain_thumb_raw_codes_to_anatomical_frgp(tmp_path: Path) -> None:
    plain_11 = tmp_path / "00001000_plain_1000_11.png"
    plain_12 = tmp_path / "00001000_plain_1000_12.png"

    row_11 = parse_file(plain_11, capture="plain", dataset="nist_sd300b", target_ppi=1000, exts=("png",))
    row_12 = parse_file(plain_12, capture="plain", dataset="nist_sd300b", target_ppi=1000, exts=("png",))

    assert row_11 is not None
    assert row_11.subject_id == 1000
    assert row_11.impression == "plain"
    assert row_11.ppi == 1000
    assert row_11.raw_frgp == 11
    assert row_11.frgp == 1

    assert row_12 is not None
    assert row_12.raw_frgp == 12
    assert row_12.frgp == 6


def test_parse_file_keeps_roll_raw_codes_as_anatomical_frgp(tmp_path: Path) -> None:
    roll_01 = tmp_path / "00001000_roll_1000_01.png"
    roll_06 = tmp_path / "00001000_roll_1000_06.png"

    row_01 = parse_file(roll_01, capture="roll", dataset="nist_sd300b", target_ppi=1000, exts=("png",))
    row_06 = parse_file(roll_06, capture="roll", dataset="nist_sd300b", target_ppi=1000, exts=("png",))

    assert row_01 is not None
    assert row_01.raw_frgp == 1
    assert row_01.frgp == 1

    assert row_06 is not None
    assert row_06.raw_frgp == 6
    assert row_06.frgp == 6


def test_build_manifest_excludes_plain_slap_and_nonexistent_roll_thumb_raw_codes(tmp_path: Path) -> None:
    plain_dir = tmp_path / "plain"
    roll_dir = tmp_path / "roll"
    for raw in (11, 12, 13, 14):
        _touch(plain_dir / f"00001000_plain_1000_{raw:02d}.png")
    for raw in (1, 6, 11, 12):
        _touch(roll_dir / f"00001000_roll_1000_{raw:02d}.png")

    manifest = build_manifest(plain_dir, roll_dir, dataset="nist_sd300b", target_ppi=1000, exts=("png",))

    plain_raw = set(manifest.loc[manifest["capture"] == "plain", "raw_frgp"].astype(int).tolist())
    roll_raw = set(manifest.loc[manifest["capture"] == "roll", "raw_frgp"].astype(int).tolist())
    assert plain_raw == {11, 12}
    assert roll_raw == {1, 6}
    assert set(manifest["frgp"].astype(int).tolist()) == {1, 6}


def test_positive_pairs_use_anatomical_thumb_mapping_without_raw_mixing(tmp_path: Path) -> None:
    plain_dir = tmp_path / "plain"
    roll_dir = tmp_path / "roll"
    for raw in (11, 12, 13, 14):
        _touch(plain_dir / f"00001000_plain_1000_{raw:02d}.png")
    for raw in (1, 6, 11, 12):
        _touch(roll_dir / f"00001000_roll_1000_{raw:02d}.png")

    manifest = build_manifest(plain_dir, roll_dir, dataset="nist_sd300b", target_ppi=1000, exts=("png",))
    manifest["split"] = "train"
    one = choose_one(manifest)
    pos = make_positive_pairs(one)

    meta = manifest[["path", "capture", "raw_frgp", "frgp"]].set_index("path")
    joined = pos.join(meta.add_prefix("a_"), on="path_a").join(meta.add_prefix("b_"), on="path_b")

    assert len(pos) == 2
    assert set(pos["frgp"].astype(int).tolist()) == {1, 6}

    by_frgp = joined.set_index("frgp")
    assert int(by_frgp.loc[1, "a_raw_frgp"]) == 11
    assert int(by_frgp.loc[1, "b_raw_frgp"]) == 1
    assert int(by_frgp.loc[6, "a_raw_frgp"]) == 12
    assert int(by_frgp.loc[6, "b_raw_frgp"]) == 6

    assert not joined["a_raw_frgp"].isin([13, 14]).any()
    assert not joined["b_raw_frgp"].isin([11, 12]).any()


def test_generate_pairs_prefers_anatomical_frgp_when_raw_frgp_is_present() -> None:
    manifest = pd.DataFrame(
        [
            {"raw_frgp": 11, "frgp": 1, "finger_id": 11},
            {"raw_frgp": 1, "frgp": 1, "finger_id": 1},
        ]
    )
    assert _infer_finger_col(manifest) == "frgp"
