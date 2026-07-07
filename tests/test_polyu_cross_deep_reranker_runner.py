"""Hermetic tests for the frozen deep-reranker PolyU Cross runner (Phase 3A.2).

These tests do not require torch, the real checkpoint, GPU, or biometric
images. The frozen-model construction and the scoring boundary are injected /
monkeypatched so the runner's own path resolution, enrichment, identity
preservation, and missing-image failure handling are exercised directly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipelines.benchmark import run_polyu_cross_deep_reranker as runner


PAIR_COLUMNS = [
    "pair_id", "label", "split", "subject_a", "subject_b",
    "finger_unit_a", "finger_unit_b", "frgp", "path_a", "path_b",
    "modality_a", "modality_b", "session_a", "session_b", "sample_uid_a", "sample_uid_b",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pair_row(pid, label, subj_a, subj_b, rel_a, rel_b):
    return {
        "pair_id": pid, "label": label, "split": "val",
        "subject_a": subj_a, "subject_b": subj_b,
        "finger_unit_a": subj_a, "finger_unit_b": subj_b, "frgp": 0,
        "path_a": rel_a, "path_b": rel_b,
        "modality_a": "contactless_2d", "modality_b": "contact_based_2d",
        "session_a": "session_1", "session_b": "session_1",
        "sample_uid_a": f"polyu_cross_a{pid}", "sample_uid_b": f"polyu_cross_b{pid}",
    }


def _make_dataset(tmp_path: Path, *, make_images: bool):
    root = tmp_path / "polyu_root"
    rows = [
        _pair_row(0, 1, 1, 1, "cl/p1/p1.bmp", "ct/1_1.jpg"),
        _pair_row(1, 0, 1, 2, "cl/p1/p2.bmp", "ct/2_1.jpg"),
        _pair_row(2, 1, 3, 3, "cl/p3/p1.bmp", "ct/3_1.jpg"),
        _pair_row(3, 0, 3, 4, "cl/p3/p2.bmp", "ct/4_1.jpg"),
    ]
    if make_images:
        for row in rows:
            for rel in (row["path_a"], row["path_b"]):
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"\x00")
    manifest_dir = tmp_path / "manifests" / "polyu_cross"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.csv").write_text("sample_uid,path\n", encoding="utf-8")
    pd.DataFrame(rows, columns=PAIR_COLUMNS).to_csv(manifest_dir / "pairs_val.csv", index=False)
    return root, manifest_dir


def _fake_frozen(tmp_path: Path) -> runner.FrozenModel:
    ckpt = tmp_path / "fake_ckpt.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    return runner.FrozenModel(
        model=object(),
        device=SimpleNamespace(type="cpu"),
        input_size=384,
        checkpoint=ckpt,
        provenance={
            "checkpoint": str(ckpt),
            "checkpoint_sha256": runner.sha256_file(ckpt),
            "model_type": "fast_ddp_pair_model_nist_only",
            "frozen": True,
            "eval_mode": True,
            "no_grad": True,
            "preprocess_identifier": runner.PREPROCESS_ID,
            "device": "cpu",
        },
    )


def _fake_scorer_factory():
    def _fake(frozen, scorable, *, batch_size=128, num_workers=0):
        n = len(scorable)
        # Deterministic, finite, and label-separated so downstream code has signal.
        probs = np.where(scorable["label"].astype(int).to_numpy() == 1, 0.8, 0.2).astype(float)
        logits = np.log(probs / (1.0 - probs))
        timing = {"total_ms": 1.0 * n, "avg_ms_pair": 1.0, "n_unique_images": 2 * n}
        return logits, probs, timing
    return _fake


def test_deep_run_preserves_identity_and_schema(tmp_path, monkeypatch):
    root, manifest_dir = _make_dataset(tmp_path, make_images=True)
    outdir = tmp_path / "out"
    monkeypatch.setattr(runner, "score_resolved_pairs", _fake_scorer_factory())

    summary = runner.run(
        checkpoint=tmp_path / "unused.pt",
        manifest_dir=manifest_dir,
        outdir=outdir,
        splits=["val"],
        limit=0,
        strict=False,
        polyu_root=str(root),
        device="cpu",
        batch_size=8,
        num_workers=0,
        frozen=_fake_frozen(tmp_path),
    )

    scores_csv = outdir / "scores_polyu_cross_deep_pair_reranker_val.csv"
    assert scores_csv.exists()
    df = pd.read_csv(scores_csv)

    # Required leading schema present and ordered first.
    assert list(df.columns)[: len(runner.REQUIRED_SCORE_COLUMNS)] == runner.REQUIRED_SCORE_COLUMNS
    src = pd.read_csv(manifest_dir / "pairs_val.csv")
    assert list(df["pair_id"]) == list(src["pair_id"])
    assert list(df["label"]) == list(src["label"])
    # PolyU identity columns preserved.
    for col in ("sample_uid_a", "sample_uid_b", "modality_a", "session_a"):
        assert list(df[col]) == list(src[col])
    assert set(df["method"]) == {runner.METHOD}
    assert set(df["dataset"]) == {"polyu_cross"}
    assert np.isfinite(pd.to_numeric(df["score"], errors="coerce")).all()
    assert (df["status"] == "ok").all()

    failures = pd.read_csv(outdir / "failures_polyu_cross_deep_pair_reranker_val.csv")
    assert failures.empty

    # Provenance recorded in run meta.
    meta = json.loads((outdir / "run_polyu_cross_deep_pair_reranker_val.meta.json").read_text(encoding="utf-8"))
    prov = meta["model_provenance"]
    assert prov["frozen"] is True
    assert prov["checkpoint_sha256"]
    assert meta["config"]["no_grad_inference"] is True
    assert meta["config"]["instantiates_optimizer_or_scheduler"] is False
    assert (outdir / "run_manifest.json").exists()
    assert (outdir / "latency_summary.csv").exists()
    assert len(summary["results"]) == 1


def test_deep_missing_images_fail_non_strict(tmp_path, monkeypatch):
    root, manifest_dir = _make_dataset(tmp_path, make_images=False)
    outdir = tmp_path / "out"
    monkeypatch.setattr(runner, "score_resolved_pairs", _fake_scorer_factory())

    runner.run(
        checkpoint=tmp_path / "unused.pt", manifest_dir=manifest_dir, outdir=outdir,
        splits=["val"], limit=0, strict=False, polyu_root=str(root),
        device="cpu", batch_size=8, num_workers=0, frozen=_fake_frozen(tmp_path),
    )
    df = pd.read_csv(outdir / "scores_polyu_cross_deep_pair_reranker_val.csv")
    assert (df["status"] == "failed").all()
    assert df["error"].str.startswith("missing_image").all()
    # Identity preserved even for failures.
    src = pd.read_csv(manifest_dir / "pairs_val.csv")
    assert list(df["pair_id"]) == list(src["pair_id"])
    assert list(df["label"]) == list(src["label"])
    failures = pd.read_csv(outdir / "failures_polyu_cross_deep_pair_reranker_val.csv")
    assert len(failures) == len(df)


def test_deep_strict_raises_on_missing(tmp_path, monkeypatch):
    root, manifest_dir = _make_dataset(tmp_path, make_images=False)
    monkeypatch.setattr(runner, "score_resolved_pairs", _fake_scorer_factory())
    with pytest.raises(runner.PolyUCrossDeepRunError):
        runner.run(
            checkpoint=tmp_path / "unused.pt", manifest_dir=manifest_dir, outdir=tmp_path / "out",
            splits=["val"], limit=0, strict=True, polyu_root=str(root),
            device="cpu", batch_size=8, num_workers=0, frozen=_fake_frozen(tmp_path),
        )


def test_deep_does_not_mutate_pairs(tmp_path, monkeypatch):
    root, manifest_dir = _make_dataset(tmp_path, make_images=True)
    monkeypatch.setattr(runner, "score_resolved_pairs", _fake_scorer_factory())
    pairs_path = manifest_dir / "pairs_val.csv"
    before = _sha256(pairs_path)
    runner.run(
        checkpoint=tmp_path / "unused.pt", manifest_dir=manifest_dir, outdir=tmp_path / "out",
        splits=["val"], limit=0, strict=False, polyu_root=str(root),
        device="cpu", batch_size=8, num_workers=0, frozen=_fake_frozen(tmp_path),
    )
    assert _sha256(pairs_path) == before
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["constraints"]["trained_or_finetuned"] is False
    assert manifest["constraints"]["ran_fusion"] is False
    assert manifest["runs"][0]["pairs_csv_sha256"] == before
