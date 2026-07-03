"""Tests for the PolyU Cross zero-shot benchmark plumbing (Phase 3A).

These tests are hermetic: they do not require OpenCV, the real evaluate.py
subprocess, the SourceAFIS sidecar, or any biometric images. The classical
scoring boundary (evaluate.py) is monkeypatched so the runner's own path
resolution, enrichment, positional join, and failure handling are exercised
directly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.fpbench.datasets import polyu_cross_pairs as pcp
from pipelines.benchmark import run_polyu_cross_zero_shot as runner


PAIR_COLUMNS = [
    "pair_id",
    "label",
    "split",
    "subject_a",
    "subject_b",
    "finger_unit_a",
    "finger_unit_b",
    "frgp",
    "path_a",
    "path_b",
    "modality_a",
    "modality_b",
    "session_a",
    "session_b",
    "sample_uid_a",
    "sample_uid_b",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pair_row(pair_id: int, label: int, split: str, subj_a: int, subj_b: int, rel_a: str, rel_b: str) -> dict:
    return {
        "pair_id": pair_id,
        "label": label,
        "split": split,
        "subject_a": subj_a,
        "subject_b": subj_b,
        "finger_unit_a": subj_a,
        "finger_unit_b": subj_b,
        "frgp": 0,
        "path_a": rel_a,
        "path_b": rel_b,
        "modality_a": "contactless_2d",
        "modality_b": "contact_based_2d",
        "session_a": "session_1",
        "session_b": "session_1",
        "sample_uid_a": f"polyu_cross_a{pair_id}",
        "sample_uid_b": f"polyu_cross_b{pair_id}",
    }


def _make_dataset_root(tmp_path: Path, *, make_images: bool) -> tuple[Path, list[dict]]:
    """Create a fake PolyU dataset root and matching pair rows (relative paths)."""

    root = tmp_path / "polyu_root"
    rows = [
        _pair_row(0, 1, "val", 1, 1, "contactless/p1/p1.bmp", "contact/1_1.jpg"),
        _pair_row(1, 0, "val", 1, 2, "contactless/p1/p2.bmp", "contact/2_1.jpg"),
        _pair_row(2, 1, "val", 3, 3, "contactless/p3/p1.bmp", "contact/3_1.jpg"),
        _pair_row(3, 0, "val", 3, 4, "contactless/p3/p2.bmp", "contact/4_1.jpg"),
    ]
    if make_images:
        for row in rows:
            for rel in (row["path_a"], row["path_b"]):
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"\x00")  # existence is all resolve_pairs_frame checks
    return root, rows


def _make_manifest_dir(tmp_path: Path, rows: list[dict], *, splits=("val",)) -> Path:
    manifest_dir = tmp_path / "manifests" / "polyu_cross"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.csv").write_text("sample_uid,path\n", encoding="utf-8")
    df = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    for split in splits:
        df[df["split"] == split].to_csv(manifest_dir / f"pairs_{split}.csv", index=False)
    return manifest_dir


# ---------------------------------------------------------------------------
# Loader / resolver unit tests
# ---------------------------------------------------------------------------
def test_load_polyu_cross_pairs_new_schema(tmp_path: Path) -> None:
    _, rows = _make_dataset_root(tmp_path, make_images=False)
    manifest_dir = _make_manifest_dir(tmp_path, rows)

    df = pcp.load_polyu_cross_pairs(manifest_dir / "pairs_val.csv")

    for column in pcp.REQUIRED_PAIR_COLUMNS:
        assert column in df.columns
    # New PolyU Cross columns survive loading.
    for column in ("sample_uid_a", "modality_a", "session_a", "finger_unit_a"):
        assert column in df.columns
    assert df["label"].dtype.kind == "i"
    assert list(df["_row_order"]) == list(range(len(df)))


def test_load_polyu_cross_pairs_missing_required_column(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"pair_id": [0], "label": [1]}).to_csv(bad, index=False)
    with pytest.raises(pcp.PolyUCrossPairError):
        pcp.load_polyu_cross_pairs(bad)


def test_resolve_root_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_dir = tmp_path / "m"
    (manifest_dir / "pairs").mkdir(parents=True)
    (manifest_dir / "pairs_split_build.meta.json").write_text(
        json.dumps({"path_base": str(tmp_path / "from_meta")}), encoding="utf-8"
    )
    (manifest_dir / "pairs" / "split_subjects.json").write_text(
        json.dumps({"resolved_data_dir": str(tmp_path / "from_split")}), encoding="utf-8"
    )

    # meta path_base wins over split resolved_data_dir
    resolved = pcp.resolve_polyu_cross_root(manifest_dir)
    assert resolved.root == (tmp_path / "from_meta")
    assert "path_base" in resolved.source

    # env var beats metadata
    monkeypatch.setenv(pcp.POLYU_CROSS_ROOT_ENV, str(tmp_path / "from_env"))
    assert pcp.resolve_polyu_cross_root(manifest_dir).root == (tmp_path / "from_env")

    # explicit override beats everything
    override = pcp.resolve_polyu_cross_root(manifest_dir, override=str(tmp_path / "from_override"))
    assert override.root == (tmp_path / "from_override")
    assert override.source == "override"


def test_resolve_pair_image_path_relative_and_absolute(tmp_path: Path) -> None:
    root = tmp_path / "root"
    # relative -> joined onto root
    assert pcp.resolve_pair_image_path("a/b.bmp", root) == root / "a" / "b.bmp"
    # POSIX absolute -> untouched
    assert pcp.resolve_pair_image_path("/abs/x.png", root) == Path("/abs/x.png")
    # Windows drive absolute -> untouched (compat with legacy bundles)
    assert pcp.looks_absolute(r"C:\data\x.png")
    # file: URI -> stripped
    assert pcp.resolve_pair_image_path("file:/C:/data/x.png", root) == Path("C:/data/x.png")
    # relative without a root is an explicit error
    with pytest.raises(pcp.PolyUCrossPairError):
        pcp.resolve_pair_image_path("a/b.bmp", None)


def test_resolve_pairs_frame_existence(tmp_path: Path) -> None:
    root, rows = _make_dataset_root(tmp_path, make_images=True)
    df = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    resolved = pcp.resolve_pairs_frame(df, root)
    assert resolved["path_a_exists"].all()
    assert resolved["path_b_exists"].all()
    # original relative paths preserved
    assert list(resolved["path_a"]) == list(df["path_a"])


# ---------------------------------------------------------------------------
# Classical runner path (evaluate.py boundary monkeypatched)
# ---------------------------------------------------------------------------
def _fake_evaluate_factory(k1: int = 10, k2: int = 10, score: float = 0.5):
    """Return a stand-in for _run_evaluate_subprocess that mimics eval_classic."""

    def _fake(*, method, split, manifest_dir, resolved_pairs_csv, work_dir):
        pairs = pd.read_csv(resolved_pairs_csv)
        raw = pd.DataFrame(
            {
                "label": pairs["label"].astype(int),
                "split": pairs["split"],
                "path_a": pairs["path_a"],
                "path_b": pairs["path_b"],
                "score": [float(score)] * len(pairs),
                "inliers": [3] * len(pairs),
                "matches": [12] * len(pairs),
                "k1": [k1] * len(pairs),
                "k2": [k2] * len(pairs),
                "extract_a_ms": [1.0] * len(pairs),
                "extract_b_ms": [1.0] * len(pairs),
                "match_ms": [0.5] * len(pairs),
                "pair_total_ms": [2.5] * len(pairs),
            }
        )
        raw_scores = Path(work_dir) / f"raw_scores_{method}_{split}.csv"
        raw.to_csv(raw_scores, index=False)
        meta = Path(work_dir) / f"eval_run_{method}_{split}.meta.json"
        meta.write_text("{}", encoding="utf-8")
        return raw_scores, meta

    return _fake


def test_classical_run_preserves_pair_id_and_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, rows = _make_dataset_root(tmp_path, make_images=True)
    manifest_dir = _make_manifest_dir(tmp_path, rows)
    outdir = tmp_path / "out"

    monkeypatch.setattr(runner, "_run_evaluate_subprocess", _fake_evaluate_factory())

    summary = runner.run(
        manifest_dir=manifest_dir,
        outdir=outdir,
        methods=["sift"],
        splits=["val"],
        limit=0,
        strict=False,
        polyu_root=str(root),
        dpi_strategy="default",
        image_dpi=None,
    )

    scores_csv = outdir / "scores_polyu_cross_sift_val.csv"
    assert scores_csv.exists()
    scored = pd.read_csv(scores_csv)

    # Required canonical schema present and ordered first.
    assert list(scored.columns)[:10] == runner.REQUIRED_SCORE_COLUMNS
    # pair_id + label preserved exactly from the source pairs.
    source = pd.read_csv(manifest_dir / "pairs_val.csv")
    assert list(scored["pair_id"]) == list(source["pair_id"])
    assert list(scored["label"]) == list(source["label"])
    assert set(scored["method"]) == {"sift"}
    assert set(scored["dataset"]) == {"polyu_cross"}
    # All images exist and features present -> all ok, no failures.
    assert (scored["status"] == "ok").all()
    failures = pd.read_csv(outdir / "failures_polyu_cross_sift_val.csv")
    assert failures.empty

    # Metadata artifacts exist.
    assert (outdir / "run_polyu_cross_sift_val.meta.json").exists()
    assert (outdir / "run_manifest.json").exists()
    assert (outdir / "latency_summary.csv").exists()
    assert len(summary["results"]) == 1


def test_runner_does_not_regenerate_or_mutate_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, rows = _make_dataset_root(tmp_path, make_images=True)
    manifest_dir = _make_manifest_dir(tmp_path, rows)
    outdir = tmp_path / "out"

    pairs_path = manifest_dir / "pairs_val.csv"
    before = _sha256(pairs_path)

    monkeypatch.setattr(runner, "_run_evaluate_subprocess", _fake_evaluate_factory())
    runner.run(
        manifest_dir=manifest_dir,
        outdir=outdir,
        methods=["sift"],
        splits=["val"],
        limit=0,
        strict=False,
        polyu_root=str(root),
        dpi_strategy="default",
        image_dpi=None,
    )

    # The canonical pairs CSV is byte-for-byte unchanged.
    assert _sha256(pairs_path) == before

    manifest = json.loads((outdir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["constraints"]["regenerated_pairs"] is False
    assert manifest["constraints"]["modified_manifest_or_pairs"] is False
    assert manifest["runs"][0]["pairs_csv_sha256"] == before


def test_missing_images_produce_failures_non_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # make_images=False -> resolved image paths will not exist
    root, rows = _make_dataset_root(tmp_path, make_images=False)
    manifest_dir = _make_manifest_dir(tmp_path, rows)
    outdir = tmp_path / "out"

    monkeypatch.setattr(runner, "_run_evaluate_subprocess", _fake_evaluate_factory())

    # Non-strict: should not raise; every pair becomes a failure.
    runner.run(
        manifest_dir=manifest_dir,
        outdir=outdir,
        methods=["sift"],
        splits=["val"],
        limit=0,
        strict=False,
        polyu_root=str(root),
        dpi_strategy="default",
        image_dpi=None,
    )
    scored = pd.read_csv(outdir / "scores_polyu_cross_sift_val.csv")
    assert (scored["status"] == "failed").all()
    assert scored["error"].str.startswith("missing_image").all()
    failures = pd.read_csv(outdir / "failures_polyu_cross_sift_val.csv")
    assert len(failures) == len(scored)
    # pair_id/label still preserved even for failed rows.
    source = pd.read_csv(manifest_dir / "pairs_val.csv")
    assert list(scored["pair_id"]) == list(source["pair_id"])
    assert list(scored["label"]) == list(source["label"])


def test_missing_images_strict_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, rows = _make_dataset_root(tmp_path, make_images=False)
    manifest_dir = _make_manifest_dir(tmp_path, rows)
    outdir = tmp_path / "out"

    monkeypatch.setattr(runner, "_run_evaluate_subprocess", _fake_evaluate_factory())
    with pytest.raises(runner.PolyUCrossRunError):
        runner.run(
            manifest_dir=manifest_dir,
            outdir=outdir,
            methods=["sift"],
            splits=["val"],
            limit=0,
            strict=True,
            polyu_root=str(root),
            dpi_strategy="default",
            image_dpi=None,
        )


def test_smoke_limit_balances_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, rows = _make_dataset_root(tmp_path, make_images=True)
    manifest_dir = _make_manifest_dir(tmp_path, rows)
    outdir = tmp_path / "out"

    monkeypatch.setattr(runner, "_run_evaluate_subprocess", _fake_evaluate_factory())
    runner.run(
        manifest_dir=manifest_dir,
        outdir=outdir,
        methods=["sift"],
        splits=["val"],
        limit=2,
        strict=False,
        polyu_root=str(root),
        dpi_strategy="default",
        image_dpi=None,
    )
    scored = pd.read_csv(outdir / "scores_polyu_cross_sift_val.csv")
    assert len(scored) == 2
    # balanced: one positive, one negative
    assert set(scored["label"]) == {0, 1}


# ---------------------------------------------------------------------------
# SourceAFIS degradation path (no sidecar in test env -> graceful failures)
# ---------------------------------------------------------------------------
def test_sourceafis_unavailable_degrades_to_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, rows = _make_dataset_root(tmp_path, make_images=True)
    manifest_dir = _make_manifest_dir(tmp_path, rows)
    outdir = tmp_path / "out"

    # Ensure the sidecar env is not enabled so the provider reports unavailable.
    monkeypatch.delenv("SOURCEAFIS_ENABLED", raising=False)
    monkeypatch.delenv("SOURCEAFIS_SERVICE_URL", raising=False)

    runner.run(
        manifest_dir=manifest_dir,
        outdir=outdir,
        methods=["sourceafis_open"],
        splits=["val"],
        limit=0,
        strict=False,
        polyu_root=str(root),
        dpi_strategy="default",
        image_dpi=None,
    )
    scores_csv = outdir / "scores_polyu_cross_sourceafis_open_val.csv"
    assert scores_csv.exists()
    scored = pd.read_csv(scores_csv)
    # No sidecar -> every pair is a graceful failure, not a crash.
    assert (scored["status"] == "failed").all()
    # Identity is still preserved for downstream joins.
    source = pd.read_csv(manifest_dir / "pairs_val.csv")
    assert list(scored["pair_id"]) == list(source["pair_id"])
    assert list(scored["label"]) == list(source["label"])
    assert set(scored["method"]) == {"sourceafis_open"}
