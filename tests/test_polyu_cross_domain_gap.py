"""Tests for the PolyU Cross domain-gap audit (Phase 4A.1).

Hermetic: builds a tiny fake PolyU Cross bundle (train/val/test) with small
deterministic images and verifies (a) TEST is never leaked into outputs and
(b) features/outputs are deterministic and byte-identical on repeat runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from scripts.diagnostics import build_polyu_cross_domain_gap as audit


PAIR_COLUMNS = [
    "pair_id", "label", "split", "subject_a", "subject_b",
    "finger_unit_a", "finger_unit_b", "frgp", "path_a", "path_b",
    "modality_a", "modality_b", "session_a", "session_b", "sample_uid_a", "sample_uid_b",
]


def _make_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = np.linspace(20, 220, 48, dtype=np.float64)
    img = np.tile(base, (48, 1))
    img = (img + rng.normal(0, 5, size=(48, 48))).clip(0, 255).astype(np.uint8)
    # Deterministic given seed (default_rng is reproducible).
    cv2.imwrite(str(path), img)


def _pair_row(pid, label, split, fu_a, fu_b, uid_a, uid_b, sess_a="session_1", sess_b="session_1"):
    return {
        "pair_id": pid, "label": label, "split": split,
        "subject_a": fu_a, "subject_b": fu_b,
        "finger_unit_a": fu_a, "finger_unit_b": fu_b, "frgp": 0,
        "path_a": f"cl/{uid_a}.png", "path_b": f"ct/{uid_b}.png",
        "modality_a": "contactless_2d", "modality_b": "contact_based_2d",
        "session_a": sess_a, "session_b": sess_b,
        "sample_uid_a": uid_a, "sample_uid_b": uid_b,
    }


def _build_bundle(tmp_path: Path):
    root = tmp_path / "polyu_root"
    rows = [
        # train genuine + impostor
        _pair_row(0, 1, "train", 1, 1, "TR_cl_1", "TR_ct_1"),
        _pair_row(1, 0, "train", 1, 2, "TR_cl_1", "TR_ct_2"),
        _pair_row(2, 1, "train", 3, 3, "TR_cl_3", "TR_ct_3", sess_a="session_2", sess_b="session_1"),
        # val genuine + impostor
        _pair_row(3, 1, "val", 5, 5, "VA_cl_5", "VA_ct_5"),
        _pair_row(4, 0, "val", 5, 6, "VA_cl_5", "VA_ct_6"),
        # TEST rows (must never appear in outputs)
        _pair_row(5, 1, "test", 9, 9, "TE_cl_9", "TE_ct_9"),
        _pair_row(6, 0, "test", 9, 8, "TE_cl_9", "TE_ct_8"),
    ]
    seed = 0
    for row in rows:
        for rel in (row["path_a"], row["path_b"]):
            _make_image(root / rel, seed)
            seed += 1
    manifest_dir = tmp_path / "manifests" / "polyu_cross"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.csv").write_text("sample_uid,path\n", encoding="utf-8")
    df = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    for split in ("train", "val", "test"):
        df[df["split"] == split].to_csv(manifest_dir / f"pairs_{split}.csv", index=False)
    return root, manifest_dir


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_no_test_leakage_and_required_finite(tmp_path):
    root, manifest_dir = _build_bundle(tmp_path)
    result = audit.run(manifest_dir=manifest_dir, outdir=tmp_path / "out", splits=["train", "val"], polyu_root=str(root))
    feats = pd.read_csv(result["features_csv"])

    uids = set(feats["sample_uid"].astype(str))
    # No TEST sample_uids anywhere.
    assert not any(u.startswith("TE_") for u in uids)
    assert set(feats["split"]) == {"train", "val"}
    # Expected TRAIN/VAL images present (deduped by sample_uid).
    assert {"TR_cl_1", "TR_ct_1", "TR_ct_2", "TR_cl_3", "TR_ct_3", "VA_cl_5", "VA_ct_5", "VA_ct_6"} == uids

    # Required basic features finite.
    for col in audit.REQUIRED_BASIC_FEATURES:
        assert np.isfinite(pd.to_numeric(feats[col], errors="coerce")).all()

    # Summary + paired outputs also free of TEST content.
    summary = pd.read_csv(result["summary_csv"])
    assert set(summary["modality"]) <= {audit.CONTACTLESS, audit.CONTACT}
    paired = pd.read_csv(result["paired_csv"])
    assert set(paired["split"]) <= {"train", "val"}


def test_audit_refuses_test_split(tmp_path):
    root, manifest_dir = _build_bundle(tmp_path)
    with pytest.raises(audit.DomainGapError):
        audit.run(manifest_dir=manifest_dir, outdir=tmp_path / "out", splits=["train", "test"], polyu_root=str(root))


def test_outputs_are_byte_identical_on_repeat(tmp_path):
    root, manifest_dir = _build_bundle(tmp_path)
    r1 = audit.run(manifest_dir=manifest_dir, outdir=tmp_path / "out1", splits=["train", "val"], polyu_root=str(root))
    r2 = audit.run(manifest_dir=manifest_dir, outdir=tmp_path / "out2", splits=["train", "val"], polyu_root=str(root))
    for key in ("features_csv", "summary_csv", "paired_csv"):
        assert _sha(r1[key]) == _sha(r2[key]), f"{key} not byte-identical across runs"


def test_pairs_not_mutated(tmp_path):
    root, manifest_dir = _build_bundle(tmp_path)
    before = {s: _sha(manifest_dir / f"pairs_{s}.csv") for s in ("train", "val", "test")}
    audit.run(manifest_dir=manifest_dir, outdir=tmp_path / "out", splits=["train", "val"], polyu_root=str(root))
    after = {s: _sha(manifest_dir / f"pairs_{s}.csv") for s in ("train", "val", "test")}
    assert before == after
