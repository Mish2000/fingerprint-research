from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from pipelines.benchmark import validate_benchmark_bundle
from pipelines.benchmark.benchmark_validation_utils import (
    BENCHMARK_CONFIG_SCHEMA_VERSION,
    BENCHMARK_RUN_META_SCHEMA_VERSION,
    validate_run_meta,
    validate_scores_csv,
)

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfeA\xe2!\xbc\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_scores_csv(path: Path) -> None:
    pd.DataFrame([
        {"label": 1, "score": 0.91},
        {"label": 0, "score": 0.12},
    ]).to_csv(path, index=False)


def _write_bundle(
    outdir: Path,
    *,
    methods: list[str],
    split: str = "val",
    showcase_claim: bool = False,
    top_level_claims: dict[str, object] | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in methods:
        scores_csv = outdir / f"scores_{method}_{split}.csv"
        roc_png = outdir / f"roc_{method}_{split}.png"
        run_meta = outdir / f"run_{method}_{split}.meta.json"
        _write_scores_csv(scores_csv)
        roc_png.write_bytes(PNG_BYTES)

        config = {
            "schema_version": BENCHMARK_CONFIG_SCHEMA_VERSION,
            "method": method,
            "split": split,
            "pairs_path": str(outdir / f"pairs_{split}.csv"),
            "manifest_path": str(outdir / "manifest.csv"),
            "resolved_data_dir": str(outdir),
        }
        if showcase_claim:
            config["showcase_eligible"] = True
            config["presentation_tier"] = "canonical"

        row = {
            "timestamp_utc": "2026-05-09T00:00:00Z",
            "method": method,
            "split": split,
            "n_pairs": 2,
            "auc": 1.0,
            "eer": 0.0,
            "tar_at_far_1e_2": 1.0,
            "tar_at_far_1e_3": 1.0,
            "avg_ms_pair_reported": 1.0,
            "avg_ms_pair_wall": 1.1,
            "scores_csv": str(scores_csv),
            "meta_json": "",
            "config_json": json.dumps(config, sort_keys=True),
        }
        if top_level_claims:
            row.update(top_level_claims)
        rows.append(row)
        run_meta.write_text(
            json.dumps(
                {
                    "schema_version": BENCHMARK_RUN_META_SCHEMA_VERSION,
                    "row": row,
                    "scores_csv": str(scores_csv),
                    "roc_png": str(roc_png),
                    "summary_csv": str(outdir / "results_summary.csv"),
                    "resolved_data_dir": config["resolved_data_dir"],
                    "manifest_path": config["manifest_path"],
                    "pairs_path": config["pairs_path"],
                    "config": config,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    pd.DataFrame(rows).to_csv(outdir / "results_summary.csv", index=False)
    (outdir / "run_manifest.json").write_text("{}", encoding="utf-8")


def test_validate_scores_csv_accepts_canonical_label_score_file(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    _write_scores_csv(scores_csv)
    df = validate_scores_csv(scores_csv, expected_n_pairs=2)
    assert list(df.columns) == ["label", "score"]


def test_validate_scores_csv_rejects_noncanonical_columns(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    pd.DataFrame([{"y_true": 1, "score": 0.91}, {"y_true": 0, "score": 0.12}]).to_csv(scores_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_scores_csv(scores_csv, expected_n_pairs=2)


def test_validate_run_meta_requires_schema_versions_and_config_alignment(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    _write_scores_csv(scores_csv)
    summary_csv = tmp_path / "results_summary.csv"
    config = {
        "schema_version": BENCHMARK_CONFIG_SCHEMA_VERSION,
        "method": "dl_quick",
        "split": "val",
        "pairs_path": "/tmp/pairs_val.csv",
        "manifest_path": "/tmp/manifest.csv",
        "resolved_data_dir": "/tmp",
        "extra": 1,
    }
    row = {
        "method": "dl_quick",
        "split": "val",
        "n_pairs": 2,
        "auc": 1.0,
        "eer": 0.0,
        "tar_at_far_1e_2": 1.0,
        "tar_at_far_1e_3": 1.0,
        "scores_csv": str(scores_csv),
        "config_json": json.dumps(config, sort_keys=True),
    }
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    run_meta = {
        "schema_version": BENCHMARK_RUN_META_SCHEMA_VERSION,
        "row": row,
        "scores_csv": str(scores_csv),
        "roc_png": str(tmp_path / "roc.png"),
        "summary_csv": str(summary_csv),
        "resolved_data_dir": "/tmp",
        "manifest_path": "/tmp/manifest.csv",
        "pairs_path": "/tmp/pairs_val.csv",
        "config": config,
    }
    run_meta_path = tmp_path / "run.meta.json"
    run_meta_path.write_text(json.dumps(run_meta), encoding="utf-8")

    payload = validate_run_meta(
        run_meta_path,
        expected_row=row,
        expected_scores_csv=scores_csv,
        expected_summary_csv=summary_csv,
        expected_method="dl_quick",
        expected_split="val",
    )
    assert payload["schema_version"] == BENCHMARK_RUN_META_SCHEMA_VERSION


def test_validate_run_meta_rejects_missing_config_schema_version(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    _write_scores_csv(scores_csv)
    summary_csv = tmp_path / "results_summary.csv"
    config = {
        "method": "dl_quick",
        "split": "val",
        "pairs_path": "/tmp/pairs_val.csv",
        "manifest_path": "/tmp/manifest.csv",
        "resolved_data_dir": "/tmp",
    }
    row = {
        "method": "dl_quick",
        "split": "val",
        "n_pairs": 2,
        "auc": 1.0,
        "eer": 0.0,
        "tar_at_far_1e_2": 1.0,
        "tar_at_far_1e_3": 1.0,
        "scores_csv": str(scores_csv),
        "config_json": json.dumps(config, sort_keys=True),
    }
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    run_meta = {
        "schema_version": BENCHMARK_RUN_META_SCHEMA_VERSION,
        "row": row,
        "scores_csv": str(scores_csv),
        "roc_png": str(tmp_path / "roc.png"),
        "summary_csv": str(summary_csv),
        "resolved_data_dir": "/tmp",
        "manifest_path": "/tmp/manifest.csv",
        "pairs_path": "/tmp/pairs_val.csv",
        "config": config,
    }
    run_meta_path = tmp_path / "run.meta.json"
    run_meta_path.write_text(json.dumps(run_meta), encoding="utf-8")

    with pytest.raises(ValueError, match="config.schema_version"):
        validate_run_meta(
            run_meta_path,
            expected_row=row,
            expected_scores_csv=scores_csv,
            expected_summary_csv=summary_csv,
            expected_method="dl_quick",
            expected_split="val",
        )


def test_validate_bundle_default_expected_methods_do_not_require_dedicated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "canonical_bundle"
    _write_bundle(outdir, methods=["classic_v2", "minutiae", "harris", "sift", "dl_quick", "vit"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_benchmark_bundle",
            "--outdir",
            str(outdir),
            "--expected_splits",
            "val",
        ],
    )

    assert validate_benchmark_bundle.main() == 0


def test_validator_expected_profiles_come_from_registry() -> None:
    assert validate_benchmark_bundle.EXPECTED_METHOD_PROFILES == {
        "canonical": ["classic_v2", "minutiae", "harris", "sift", "dl_quick", "vit"],
        "research": [
            "classic_v2",
            "minutiae",
            "harris",
            "sift",
            "dl_quick",
            "vit",
            "sift_plain_roll_v2",
            "dedicated",
        ],
        "dedicated": ["dedicated"],
    }
    assert "dedicated" not in validate_benchmark_bundle.EXPECTED_METHOD_PROFILES["canonical"]


def test_validate_bundle_research_profile_can_expect_dedicated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "research_bundle"
    _write_bundle(
        outdir,
        methods=[
            "classic_v2",
            "minutiae",
            "harris",
            "sift",
            "dl_quick",
            "vit",
            "sift_plain_roll_v2",
            "dedicated",
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_benchmark_bundle",
            "--outdir",
            str(outdir),
            "--profile",
            "research",
            "--expected_splits",
            "val",
        ],
    )

    assert validate_benchmark_bundle.main() == 0


def test_validate_bundle_rejects_dedicated_showcase_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outdir = tmp_path / "bad_research_bundle"
    _write_bundle(outdir, methods=["dedicated"], showcase_claim=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_benchmark_bundle",
            "--outdir",
            str(outdir),
            "--profile",
            "dedicated",
            "--expected_splits",
            "val",
        ],
    )

    with pytest.raises(SystemExit, match="row claims showcase/canonical eligibility"):
        validate_benchmark_bundle.main()


@pytest.mark.parametrize(
    "top_level_claims",
    [
        {"showcase_eligible": True},
        {"presentation_tier": "canonical"},
        {"canonical_default": True},
        {"benchmark_default": True},
    ],
)
def test_validate_bundle_rejects_dedicated_top_level_showcase_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    top_level_claims: dict[str, object],
) -> None:
    outdir = tmp_path / "bad_top_level_research_bundle"
    _write_bundle(outdir, methods=["dedicated"], top_level_claims=top_level_claims)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_benchmark_bundle",
            "--outdir",
            str(outdir),
            "--profile",
            "dedicated",
            "--expected_splits",
            "val",
        ],
    )

    with pytest.raises(SystemExit, match="row claims showcase/canonical eligibility"):
        validate_benchmark_bundle.main()
