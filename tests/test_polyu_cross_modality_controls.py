"""Tests for PolyU Cross modality-control pair construction (Phase 4A.1B).

Hermetic: exercises deterministic within-modality pair generation and its
sanity checks on a tiny fake manifest. No images, torch, or sidecar required.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.diagnostics import run_polyu_cross_modality_controls as mc


def _fake_manifest(tmp_path: Path) -> Path:
    rows = []

    def add(split, fu, modality, uid_prefix):
        for ses, n in (("session_1", 2), ("session_2", 2)):
            for k in range(n):
                uid = f"{uid_prefix}_{fu}_{modality[:2]}_{ses[-1]}_{k}"
                rows.append(
                    {
                        "finger_unit_id": fu,
                        "sample_uid": uid,
                        "modality": modality,
                        "session_id": ses,
                        "split": split,
                        "path": f"{split}/{modality}/{uid}.png",
                    }
                )

    for fu in (1, 2, 3, 4):
        for mod in (mc.CONTACT, mc.CONTACTLESS):
            add("train", fu, mod, "TR")
    for fu in (10, 11):
        for mod in (mc.CONTACT, mc.CONTACTLESS):
            add("val", fu, mod, "VA")
    for fu in (99,):
        for mod in (mc.CONTACT, mc.CONTACTLESS):
            add("test", fu, mod, "TE")

    manifest_dir = tmp_path / "manifests" / "polyu_cross"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_dir / "manifest.csv", index=False)
    return manifest_dir


def _test_uids(manifest_dir: Path) -> set[str]:
    m = pd.read_csv(manifest_dir / "manifest.csv", dtype=str)
    return set(m[m["split"] == "test"]["sample_uid"].astype(str))


@pytest.mark.parametrize("protocol_id,modality,relation", mc.CONTROL_PROTOCOLS)
def test_build_and_sanity(tmp_path, protocol_id, modality, relation):
    manifest_dir = _fake_manifest(tmp_path)
    images = mc.load_manifest_images(manifest_dir)
    assert set(images["split"]) == {"train", "val"}  # TEST filtered out at load

    df = mc.build_control_pairs(
        images, protocol_id=protocol_id, modality=modality, relation=relation,
        split="train", max_pos=50, neg_per_pos=3, base_seed=42,
    )
    report = mc.sanity_check_pairs(df, modality=modality, relation=relation, split="train", test_uids=_test_uids(manifest_dir))
    assert report["positives_same_finger_unit"]
    assert report["negatives_diff_finger_unit"]
    assert report["no_self_pairs"]
    assert report["no_duplicate_unordered_pairs"]
    assert report["modality_constraint"]
    assert report["session_constraint"]
    assert report["no_test_samples"]
    # No TEST uid leaked.
    assert not any(str(u).startswith("TE_") for u in df["sample_uid_a"]).__bool__()
    assert not set(df["sample_uid_a"]).intersection(_test_uids(manifest_dir))
    assert not set(df["sample_uid_b"]).intersection(_test_uids(manifest_dir))


def test_cross_session_relation_holds(tmp_path):
    manifest_dir = _fake_manifest(tmp_path)
    images = mc.load_manifest_images(manifest_dir)
    df = mc.build_control_pairs(
        images, protocol_id="contactless_to_contactless_cross_session", modality=mc.CONTACTLESS,
        relation="cross", split="train", max_pos=50, neg_per_pos=3, base_seed=42,
    )
    assert (df["session_a"].astype(str) != df["session_b"].astype(str)).all()


def test_deterministic_regeneration(tmp_path):
    manifest_dir = _fake_manifest(tmp_path)
    images = mc.load_manifest_images(manifest_dir)
    kw = dict(protocol_id="contact_based_to_contact_based_same_session", modality=mc.CONTACT,
              relation="same", split="train", max_pos=50, neg_per_pos=3, base_seed=42)
    a = mc.build_control_pairs(images, **kw)
    b = mc.build_control_pairs(images, **kw)
    pd.testing.assert_frame_equal(a, b)


def test_refuses_negative_ratio_and_labels(tmp_path):
    manifest_dir = _fake_manifest(tmp_path)
    images = mc.load_manifest_images(manifest_dir)
    df = mc.build_control_pairs(
        images, protocol_id="contact_based_to_contact_based_same_session", modality=mc.CONTACT,
        relation="same", split="val", max_pos=50, neg_per_pos=3, base_seed=42,
    )
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    assert n_pos > 0 and n_neg > 0
    # Roughly the 3:1 negative ratio (bounded by availability).
    assert n_neg <= 3 * n_pos
