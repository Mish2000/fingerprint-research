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
    RetryConfig,
    SourceAfisBenchmarkError,
    build_metrics_table,
    build_parser,
    build_threshold_table,
    call_with_retries,
    compute_auc_eer,
    compute_confusion,
    ensure_sourceafis_available,
    file_sha256,
    infer_dpi_from_path,
    load_plain_roll_pairs,
    output_schema,
    resolve_image_dpi,
    run_benchmark,
    validate_dataset_dpi,
)
from src.fpbench.fingerprint_engine.errors import ProviderUnavailableError, TemplateExtractionError
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


class RecordingSourceAfisEngine(MockSourceAfisEngine):
    def __init__(self) -> None:
        self.images: list[FingerprintImage] = []

    def extract_template(self, image: FingerprintImage) -> FingerprintTemplate:
        self.images.append(image)
        return super().extract_template(image)


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


def _write_sd300_dpi_pair_fixture(repo_root: Path, split: str = "val") -> None:
    manifest_dir = repo_root / "data" / "manifests" / "nist_sd300b"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_dir = repo_root / "data" / "raw" / "NIST" / "sd300b" / "images" / "1000" / "png"
    plain = image_dir / "plain_0.png"
    roll = image_dir / "roll_0.png"
    image_dir.mkdir(parents=True, exist_ok=True)
    plain.write_bytes(b"plain")
    roll.write_bytes(b"roll")
    pd.DataFrame(
        [
            {
                "pair_id": "1",
                "label": 1,
                "split": split,
                "subject_a": "s1",
                "subject_b": "s1",
                "frgp": 1,
                "path_a": str(plain),
                "path_b": str(roll),
            }
        ]
    ).to_csv(manifest_dir / f"pairs_{split}.csv", index=False)


def test_infer_from_path_extracts_1000_for_sd300_paths() -> None:
    assert infer_dpi_from_path(r"data\raw\NIST\sd300b\images\1000\png\plain.png") == 1000
    assert infer_dpi_from_path("data/raw/NIST/sd300c/images/2000/png/roll.png") == 2000


def test_explicit_dpi_overrides_path_inference() -> None:
    path = "data/raw/NIST/sd300b/images/1000/png/plain.png"

    assert resolve_image_dpi(path, dpi_strategy="explicit", image_dpi=500) == 500


def test_default_dpi_strategy_resolves_no_dpi() -> None:
    path = "data/raw/NIST/sd300b/images/1000/png/plain.png"

    assert resolve_image_dpi(path, dpi_strategy="default", image_dpi=1000) is None


def test_nist_dpi_validation_requires_known_dpi_for_inferred_strategy() -> None:
    pairs = pd.DataFrame(
        [
            {
                "path_a": "data/raw/NIST/sd300b/plain_0.png",
                "path_b": "data/raw/NIST/sd300b/roll_0.png",
            }
        ]
    )

    with pytest.raises(SourceAfisBenchmarkError, match="requires 1000 DPI"):
        validate_dataset_dpi(
            pairs,
            dataset="nist_sd300b",
            dpi_strategy="infer_from_path",
            image_dpi=None,
        )


def test_nist_dpi_validation_rejects_sd300c_mismatched_inferred_dpi() -> None:
    pairs = pd.DataFrame(
        [
            {
                "path_a": "data/raw/NIST/sd300c/images/1000/png/plain_0.png",
                "path_b": "data/raw/NIST/sd300c/images/1000/png/roll_0.png",
            }
        ]
    )

    with pytest.raises(SourceAfisBenchmarkError, match="requires 2000 DPI"):
        validate_dataset_dpi(
            pairs,
            dataset="nist_sd300c",
            dpi_strategy="infer_from_path",
            image_dpi=None,
        )


def test_default_dpi_strategy_is_explicitly_permissive_for_nist() -> None:
    pairs = pd.DataFrame(
        [
            {
                "path_a": "data/raw/NIST/sd300b/plain_0.png",
                "path_b": "data/raw/NIST/sd300b/roll_0.png",
            }
        ]
    )

    validate_dataset_dpi(
        pairs,
        dataset="nist_sd300b",
        dpi_strategy="default",
        image_dpi=None,
    )


def test_benchmark_includes_inferred_dpi_in_template_extraction_request(tmp_path: Path) -> None:
    _write_sd300_dpi_pair_fixture(tmp_path)
    engine = RecordingSourceAfisEngine()

    paths = run_benchmark(
        datasets=("nist_sd300b",),
        splits=("val",),
        outdir=tmp_path / "reports",
        target_fars=(0.5,),
        engine=engine,
        require_enabled_env=False,
        repo_root=tmp_path,
        template_cache_dir=tmp_path / "cache",
    )

    scores = pd.read_csv(paths["scores_val"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert {image.dpi for image in engine.images} == {1000}
    assert {image.metadata["dpi"] for image in engine.images} == {1000}
    assert scores["dpi_a"].tolist() == [1000]
    assert scores["dpi_b"].tolist() == [1000]
    assert manifest["dpi_handling"]["dpi_strategy"] == "infer_from_path"
    assert manifest["dpi_handling"]["inferred_dpi_counts"] == {"1000": 2}
    assert manifest["dpi_handling"]["unknown_dpi_image_count"] == 0


def test_default_dpi_strategy_sends_no_dpi_to_template_extraction(tmp_path: Path) -> None:
    _write_sd300_dpi_pair_fixture(tmp_path)
    engine = RecordingSourceAfisEngine()

    paths = run_benchmark(
        datasets=("nist_sd300b",),
        splits=("val",),
        outdir=tmp_path / "reports",
        target_fars=(0.5,),
        engine=engine,
        require_enabled_env=False,
        repo_root=tmp_path,
        template_cache_dir=tmp_path / "cache",
        dpi_strategy="default",
        image_dpi=1000,
    )

    scores = pd.read_csv(paths["scores_val"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert {image.dpi for image in engine.images} == {None}
    assert all("dpi" not in image.metadata for image in engine.images)
    assert scores["dpi_a"].isna().tolist() == [True]
    assert scores["dpi_b"].isna().tolist() == [True]
    assert manifest["dpi_handling"]["sidecar_default_dpi_note"]
    assert "not the main validated NIST result" in manifest["dpi_handling"]["sidecar_default_dpi_note"]


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
    assert manifest["timeout_settings"]["extract_timeout_seconds"] == pytest.approx(120.0)
    assert manifest["retry_settings"]["max_retries"] == 1
    assert manifest["sample_strategy"] == "balanced_spread"
    assert manifest["sample_seed"] == 13
    assert manifest["dpi_handling"]["dpi_strategy"] == "infer_from_path"
    assert manifest["dpi_handling"]["image_dpi"] is None
    assert manifest["dpi_handling"]["unknown_dpi_image_count"] == 16
    assert manifest["sidecar_warmup"]["ok"] is True
    assert "template_base64" not in summary
    assert "## DPI Handling" in summary
    assert "sourceafis_plain_roll_scores_val.csv" in summary


def test_selected_pairs_dir_is_scored_exactly_without_resampling(tmp_path: Path) -> None:
    _write_pair_fixture(tmp_path, "val")
    source_pairs = pd.read_csv(tmp_path / "data" / "manifests" / "toy" / "pairs_val.csv")
    selected = source_pairs.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    selected_dir = tmp_path / "selected_pairs"
    selected_dir.mkdir()
    selected_path = selected_dir / "pairs_toy_val.csv"
    selected.to_csv(selected_path, index=False)

    paths = run_benchmark(
        datasets=("toy",),
        splits=("val",),
        outdir=tmp_path / "reports" / "sourceafis",
        target_fars=(0.50,),
        engine=MockSourceAfisEngine(),
        require_enabled_env=False,
        repo_root=tmp_path,
        template_cache_dir=tmp_path / "cache",
        selected_pairs_dir=selected_dir,
    )

    scores_val = pd.read_csv(paths["scores_val"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    status = manifest["datasets"][0]

    assert scores_val["pair_id"].astype(str).tolist() == selected["pair_id"].astype(str).tolist()
    assert scores_val["label"].astype(int).tolist() == selected["label"].astype(int).tolist()
    assert status["source_is_selected_pairs"] is True
    assert status["selected_pairs_csv"] == str(selected_path)
    assert status["selected_pairs_row_count"] == len(selected)
    assert status["selected_pairs_sha256"] == file_sha256(selected_path)


def test_cli_parses_runtime_hardening_args() -> None:
    args = build_parser().parse_args(
        [
            "--request_timeout_seconds",
            "30",
            "--extract_timeout_seconds",
            "90",
            "--verify_timeout_seconds",
            "45",
            "--max_retries",
            "3",
            "--retry_backoff_seconds",
            "0.25",
            "--sample_strategy",
            "balanced_spread",
            "--sample_seed",
            "21",
            "--dpi_strategy",
            "explicit",
            "--image_dpi",
            "1000",
            "--selected_pairs_dir",
            "artifacts/reports/benchmark/plain_roll_final_baselines_v1/selected_pairs",
        ]
    )

    assert args.request_timeout_seconds == pytest.approx(30.0)
    assert args.extract_timeout_seconds == pytest.approx(90.0)
    assert args.verify_timeout_seconds == pytest.approx(45.0)
    assert args.max_retries == 3
    assert args.retry_backoff_seconds == pytest.approx(0.25)
    assert args.sample_strategy == "balanced_spread"
    assert args.sample_seed == 21
    assert args.dpi_strategy == "explicit"
    assert args.image_dpi == 1000
    assert args.selected_pairs_dir == "artifacts/reports/benchmark/plain_roll_final_baselines_v1/selected_pairs"


def test_balanced_spread_sampling_is_deterministic_and_spread(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "data" / "manifests" / "toy"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx in range(40):
        label = 1 if idx % 2 else 0
        rows.append(
            {
                "pair_id": str(idx),
                "label": label,
                "split": "val",
                "subject_a": f"s{idx}",
                "subject_b": f"s{idx}" if label else f"other-{idx}",
                "frgp": idx % 10,
                "path_a": f"C:/fingerprint-research/plain_{idx}.png",
                "path_b": f"C:/fingerprint-research/roll_{idx}.png",
            }
        )
    pd.DataFrame(rows).to_csv(manifest_dir / "pairs_val.csv", index=False)

    sampled_a, _ = load_plain_roll_pairs(
        "toy",
        "val",
        repo_root=tmp_path,
        limit=10,
        sample_strategy="balanced_spread",
        sample_seed=7,
    )
    sampled_b, _ = load_plain_roll_pairs(
        "toy",
        "val",
        repo_root=tmp_path,
        limit=10,
        sample_strategy="balanced_spread",
        sample_seed=7,
    )
    first, _ = load_plain_roll_pairs(
        "toy",
        "val",
        repo_root=tmp_path,
        limit=10,
        sample_strategy="first",
        sample_seed=7,
    )

    assert sampled_a["pair_id"].tolist() == sampled_b["pair_id"].tolist()
    assert int((sampled_a["label"] == 1).sum()) == 5
    assert int((sampled_a["label"] == 0).sum()) == 5
    assert sampled_a["pair_id"].astype(int).max() > first["pair_id"].astype(int).max()


def test_retry_only_transient_transport_failures() -> None:
    attempts = {"transient": 0, "invalid": 0}

    def transient_then_ok() -> str:
        attempts["transient"] += 1
        if attempts["transient"] == 1:
            raise ProviderUnavailableError("SourceAFIS template extraction timed out contacting sidecar.")
        return "ok"

    result = call_with_retries("extract_template", transient_then_ok, RetryConfig(max_retries=2, retry_backoff_seconds=0))

    assert result.value == "ok"
    assert result.retry_count == 1
    assert attempts["transient"] == 2

    def invalid_image() -> str:
        attempts["invalid"] += 1
        raise TemplateExtractionError("image decode failed")

    with pytest.raises(TemplateExtractionError):
        call_with_retries("extract_template", invalid_image, RetryConfig(max_retries=2, retry_backoff_seconds=0))
    assert attempts["invalid"] == 1


class FailingExtractionEngine(MockSourceAfisEngine):
    def __init__(self) -> None:
        self.extract_calls_by_path: dict[str, int] = {}

    def extract_template(self, image: FingerprintImage) -> FingerprintTemplate:
        path = str(image.path)
        self.extract_calls_by_path[path] = self.extract_calls_by_path.get(path, 0) + 1
        raise TemplateExtractionError("image decode failed")


def test_repeated_image_extraction_failure_is_cached_in_run(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "data" / "manifests" / "toy"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    shared_plain = image_dir / "plain_shared.png"
    shared_plain.write_bytes(b"not-a-real-png")
    for idx in range(2):
        (image_dir / f"roll_{idx}.png").write_bytes(f"roll-{idx}".encode("ascii"))
    pd.DataFrame(
        [
            {
                "pair_id": "0",
                "label": 0,
                "split": "val",
                "subject_a": "s1",
                "subject_b": "s2",
                "frgp": 1,
                "path_a": str(shared_plain),
                "path_b": str(image_dir / "roll_0.png"),
            },
            {
                "pair_id": "1",
                "label": 1,
                "split": "val",
                "subject_a": "s1",
                "subject_b": "s1",
                "frgp": 1,
                "path_a": str(shared_plain),
                "path_b": str(image_dir / "roll_1.png"),
            },
        ]
    ).to_csv(manifest_dir / "pairs_val.csv", index=False)
    engine = FailingExtractionEngine()

    paths = run_benchmark(
        datasets=("toy",),
        splits=("val",),
        outdir=tmp_path / "reports",
        target_fars=(0.5,),
        engine=engine,
        require_enabled_env=False,
        repo_root=tmp_path,
        template_cache_dir=tmp_path / "cache",
        retry_backoff_seconds=0,
    )

    failures = pd.read_csv(paths["failures"])
    scores = pd.read_csv(paths["scores_val"])
    metrics = pd.read_csv(paths["metrics"])

    assert engine.extract_calls_by_path[str(shared_plain)] == 1
    assert failures["cached_failure"].tolist() == [False, True]
    assert list(scores["raw_score"].isna()) == [True, True]
    assert int(metrics.iloc[0]["n_unscored"]) == 2
