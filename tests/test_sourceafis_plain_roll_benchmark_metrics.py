from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.diagnostics.run_sourceafis_plain_roll_benchmark import (
    FAILURE_COLUMNS,
    METRICS_COLUMNS,
    SCORES_COLUMNS,
    THRESHOLD_COLUMNS,
    SourceAfisBenchmarkError,
    build_metrics_table,
    build_threshold_table,
    compute_auc_eer,
    compute_confusion,
    ensure_sourceafis_available,
    output_schema,
    run_benchmark,
)
from src.fpbench.fingerprint_engine.types import (
    EngineCapabilities,
    EngineMetadata,
    FingerprintImage,
    FingerprintTemplate,
    GalleryTemplate,
    IdentificationResult,
    MatchResult,
    QualityResult,
)


def _metadata(*, available: bool = True, reason: str | None = None) -> EngineMetadata:
    return EngineMetadata(
        provider_id="sourceafis_open",
        provider_version="mock-sourceafis-1",
        name="Mock SourceAFIS",
        description="Mock SourceAFIS provider for benchmark unit tests.",
        available=available,
        unavailable_reason=reason,
        template_format="sourceafis",
        template_version="mock-sourceafis-1",
        sdk_name="SourceAFIS",
        capabilities=EngineCapabilities(
            supports_template_extraction=True,
            supports_verification=True,
            supports_identification=False,
            supports_quality=False,
            supports_template_storage=True,
            template_formats=["sourceafis"],
            normalized_score_range=None,
        ),
        metadata={"service_url": "mock://sourceafis"},
    )


class UnavailableMockEngine:
    def metadata(self) -> EngineMetadata:
        return _metadata(available=False, reason="mock sidecar is offline")

    def extract_template(self, image: FingerprintImage) -> FingerprintTemplate:
        raise AssertionError("unavailable engine should not extract")

    def verify(
        self,
        probe_template: FingerprintTemplate,
        candidate_template: FingerprintTemplate,
    ) -> MatchResult:
        raise AssertionError("unavailable engine should not verify")

    def identify(
        self,
        probe_template: FingerprintTemplate,
        gallery: list[GalleryTemplate],
        top_k: int = 10,
    ) -> IdentificationResult:
        raise AssertionError("unavailable engine should not identify")

    def assess_quality(self, image: FingerprintImage) -> QualityResult | None:
        return None


class MockSourceAfisEngine:
    scores = {
        ("val", "0"): 0.10,
        ("val", "1"): 0.90,
        ("val", "2"): 0.30,
        ("val", "3"): 0.80,
        ("test", "0"): 0.20,
        ("test", "1"): 0.70,
        ("test", "2"): 0.60,
        ("test", "3"): 0.95,
    }

    def metadata(self) -> EngineMetadata:
        return _metadata()

    def extract_template(self, image: FingerprintImage) -> FingerprintTemplate:
        assert image.image_bytes
        return FingerprintTemplate(
            provider_id="sourceafis_open",
            provider_version="mock-sourceafis-1",
            template_format="sourceafis",
            template_version="mock-sourceafis-1",
            template_bytes=f"template:{image.image_id}".encode("ascii"),
            image_id=image.image_id,
            metadata={"fixture": "mock"},
        )

    def verify(
        self,
        probe_template: FingerprintTemplate,
        candidate_template: FingerprintTemplate,
    ) -> MatchResult:
        del candidate_template
        parts = str(probe_template.image_id).split(":")
        split = parts[1]
        pair_id = parts[2]
        return MatchResult(
            provider_id="sourceafis_open",
            provider_version="mock-sourceafis-1",
            score=self.scores[(split, pair_id)],
            normalized_score=None,
            latency_ms=1.25,
            metadata={"fixture": "mock"},
        )

    def identify(
        self,
        probe_template: FingerprintTemplate,
        gallery: list[GalleryTemplate],
        top_k: int = 10,
    ) -> IdentificationResult:
        del probe_template, gallery, top_k
        return IdentificationResult(provider_id="sourceafis_open", provider_version="mock-sourceafis-1")

    def assess_quality(self, image: FingerprintImage) -> QualityResult | None:
        del image
        return None


def test_tar_far_frr_auc_and_eer_calculations() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.90, 0.80, 0.20, 0.10]

    counts = compute_confusion(labels, scores, 0.50)
    auc, eer, eer_threshold = compute_auc_eer(labels, scores)

    assert counts["tar"] == 1.0
    assert counts["far"] == 0.0
    assert counts["frr"] == 0.0
    assert counts["ta"] == 2
    assert counts["fa"] == 0
    assert auc == 1.0
    assert eer == 0.0
    assert eer_threshold >= 0.20


def test_auc_and_eer_are_deterministic_for_complete_ties() -> None:
    auc, eer, threshold = compute_auc_eer([1, 0], [0.50, 0.50])

    assert auc == 0.5
    assert eer == 0.5
    assert threshold == pytest.approx(0.50)


def test_metrics_tables_apply_val_thresholds_to_test() -> None:
    scores = pd.DataFrame(
        [
            {"dataset": "toy", "split": "val", "pair_id": "0", "label": 0, "raw_score": 0.10},
            {"dataset": "toy", "split": "val", "pair_id": "1", "label": 0, "raw_score": 0.40},
            {"dataset": "toy", "split": "val", "pair_id": "2", "label": 0, "raw_score": 0.80},
            {"dataset": "toy", "split": "val", "pair_id": "3", "label": 1, "raw_score": 0.50},
            {"dataset": "toy", "split": "test", "pair_id": "0", "label": 0, "raw_score": 0.70},
            {"dataset": "toy", "split": "test", "pair_id": "1", "label": 1, "raw_score": 0.60},
        ]
    )

    thresholds = build_threshold_table(scores, (0.50,))
    metrics = build_metrics_table(scores, thresholds)
    test_metric = metrics[metrics["split"] == "test"].iloc[0]

    assert list(thresholds.columns) == THRESHOLD_COLUMNS
    assert list(metrics.columns) == METRICS_COLUMNS
    assert thresholds.iloc[0]["threshold"] == pytest.approx(0.50)
    assert test_metric["threshold"] == pytest.approx(0.50)
    assert test_metric["tar"] == 1.0
    assert test_metric["far"] == 1.0


def test_unavailable_provider_fails_before_benchmark_work() -> None:
    with pytest.raises(SourceAfisBenchmarkError, match="mock sidecar is offline"):
        ensure_sourceafis_available(UnavailableMockEngine(), require_enabled_env=False)


def _write_pair_fixture(repo_root: Path, split: str) -> None:
    manifest_dir = repo_root / "data" / "manifests" / "toy"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_dir = repo_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        (image_dir / f"plain_{split}_{idx}.png").write_bytes(f"plain-{split}-{idx}".encode("ascii"))
        (image_dir / f"roll_{split}_{idx}.png").write_bytes(f"roll-{split}-{idx}".encode("ascii"))

    rows: list[dict[str, Any]] = []
    for idx, label in enumerate([0, 1, 0, 1]):
        subject_a = "s1" if idx < 2 else "s2"
        subject_b = subject_a if label == 1 else f"other-{idx}"
        rows.append(
            {
                "pair_id": str(idx),
                "label": label,
                "split": split,
                "subject_a": subject_a,
                "subject_b": subject_b,
                "frgp": 1,
                "path_a": str(image_dir / f"plain_{split}_{idx}.png"),
                "path_b": str(image_dir / f"roll_{split}_{idx}.png"),
            }
        )
    pd.DataFrame(rows).to_csv(manifest_dir / f"pairs_{split}.csv", index=False)


def test_mocked_provider_generates_output_schema_without_sidecar(tmp_path: Path) -> None:
    _write_pair_fixture(tmp_path, "val")
    _write_pair_fixture(tmp_path, "test")

    paths = run_benchmark(
        datasets=("toy",),
        splits=("val", "test"),
        outdir=tmp_path / "reports" / "sourceafis",
        target_fars=(0.50,),
        engine=MockSourceAfisEngine(),
        require_enabled_env=False,
        repo_root=tmp_path,
        template_cache_dir=tmp_path / "cache",
    )

    scores_val = pd.read_csv(paths["scores_val"])
    scores_test = pd.read_csv(paths["scores_test"])
    thresholds = pd.read_csv(paths["thresholds"])
    metrics = pd.read_csv(paths["metrics"])
    failures = pd.read_csv(paths["failures"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    summary = paths["summary"].read_text(encoding="utf-8")

    assert list(scores_val.columns) == SCORES_COLUMNS
    assert list(scores_test.columns) == SCORES_COLUMNS
    assert list(thresholds.columns) == THRESHOLD_COLUMNS
    assert list(metrics.columns) == METRICS_COLUMNS
    assert list(failures.columns) == FAILURE_COLUMNS
    assert failures.empty
    assert thresholds.iloc[0]["threshold"] == pytest.approx(0.30)
    assert metrics[(metrics["split"] == "test") & (metrics["target_far"] == 0.50)].iloc[0]["tar"] == 1.0
    assert manifest["schema_version"] == "sourceafis_open_plain_roll_benchmark_v1"
    assert manifest["output_schema"] == output_schema()
    assert "template_base64" not in summary
    assert "sourceafis_plain_roll_scores_val.csv" in summary
