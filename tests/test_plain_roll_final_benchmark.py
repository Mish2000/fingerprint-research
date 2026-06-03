from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipelines.benchmark import run_plain_roll_final_benchmark as final
from scripts.diagnostics import run_sourceafis_plain_roll_benchmark as sourceafis


def _write_dataset(repo: Path, dataset: str, rows_by_split: dict[str, list[dict[str, Any]]]) -> Path:
    dataset_dir = repo / "data" / "manifests" / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.csv").write_text("path\n", encoding="utf-8")
    for split, rows in rows_by_split.items():
        pd.DataFrame(rows).to_csv(dataset_dir / f"pairs_{split}.csv", index=False)
    return dataset_dir


def _pair(pair_id: str, label: int, split: str, score: float | None = None) -> dict[str, Any]:
    subject_a = f"subject-{pair_id}"
    subject_b = subject_a if int(label) == 1 else f"other-{pair_id}"
    return {
        "pair_id": pair_id,
        "label": int(label),
        "split": split,
        "subject_a": subject_a,
        "subject_b": subject_b,
        "frgp": "3",
        "path_a": f"C:/fingerprints/{split}/plain_{pair_id}.png",
        "path_b": f"C:/fingerprints/{split}/rolled_{pair_id}.png",
        "fixture_score": score,
    }


def _existing_pair(
    tmp_path: Path,
    pair_id: str,
    label: int,
    split: str = "val",
    *,
    subject_a: str | None = None,
    subject_b: str | None = None,
    finger: str = "3",
    path_order: str = "plain_roll",
) -> dict[str, Any]:
    image_dir = tmp_path / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    plain = image_dir / f"plain_{pair_id}.png"
    rolled = image_dir / f"rolled_{pair_id}.png"
    plain.write_bytes(b"plain")
    rolled.write_bytes(b"rolled")
    if path_order == "roll_plain":
        path_a, path_b = rolled, plain
    elif path_order == "plain_plain":
        second_plain = image_dir / f"plain_second_{pair_id}.png"
        second_plain.write_bytes(b"plain2")
        path_a, path_b = plain, second_plain
    else:
        path_a, path_b = plain, rolled

    resolved_subject_a = subject_a or f"subject-{pair_id}"
    if subject_b is None:
        subject_b = resolved_subject_a if int(label) == 1 else f"other-{pair_id}"
    return {
        "pair_id": pair_id,
        "label": int(label),
        "split": split,
        "subject_a": resolved_subject_a,
        "subject_b": subject_b,
        "frgp": finger,
        "path_a": str(path_a),
        "path_b": str(path_b),
    }


def _audit_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return final.audit_pair_dataframe(
        pd.DataFrame(rows),
        dataset="toy",
        split="val",
        selected_pairs_csv=tmp_path / "selected.csv",
        repo_root=tmp_path,
    )


def _install_fake_evaluator(monkeypatch: pytest.MonkeyPatch, score_by_pair: dict[str, float]) -> list[list[str]]:
    calls: list[list[str]] = []

    def value(cmd: list[str], flag: str) -> str:
        return cmd[cmd.index(flag) + 1]

    def fake_run(cmd: list[str], cwd: str | None = None, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del cwd, kwargs
        calls.append([str(item) for item in cmd])
        method = value(calls[-1], "--method")
        pairs_file = Path(value(calls[-1], "--pairs_file"))
        out_scores = Path(value(calls[-1], "--out_scores"))
        out_run_meta = Path(value(calls[-1], "--out_run_meta"))
        pairs = pd.read_csv(pairs_file)
        scores = pairs[["pair_id", "label", "split", "path_a", "path_b"]].copy()
        scores["score"] = [float(score_by_pair[str(pair_id)]) for pair_id in scores["pair_id"]]

        if method in {"classic_v2", "harris", "sift", "sift_plain_roll_v2"}:
            scores["extract_a_ms"] = 1.0
            scores["extract_b_ms"] = 2.0
            scores["match_ms"] = 3.0
            scores["pair_total_ms"] = 6.0
            meta_path = Path(str(out_scores) + ".meta.json")
            meta_payload = {
                "avg_ms_pair": 6.0,
                "p50_ms_pair": 6.0,
                "p95_ms_pair": 6.0,
                "total_ms": 6.0 * len(scores),
                "feature_cache": {"hits": 2, "misses": 3},
            }
        elif method == "minutiae":
            scores["pair_total_ms"] = 9.0
            meta_path = out_scores.with_suffix(".meta.json")
            meta_payload = {
                "avg_ms_pair": 9.0,
                "p50_ms_pair": 9.0,
                "p95_ms_pair": 9.0,
                "total_ms": 9.0 * len(scores),
                "cache_hits": 4,
                "cache_misses": 5,
                "template_cache": {"hits": 4, "misses": 5},
            }
        else:
            meta_path = out_scores.with_suffix(".meta.json")
            meta_payload = {"avg_ms_pair": 12.0}

        out_scores.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out_scores, index=False)
        meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")
        out_run_meta.write_text(
            json.dumps(
                {
                    "timing": {
                        "avg_ms_pair_reported": meta_payload["avg_ms_pair"],
                        "avg_ms_pair_wall": meta_payload["avg_ms_pair"] + 1.0,
                    },
                    "method_meta_json": str(meta_path),
                    "row": {
                        "avg_ms_pair_reported": meta_payload["avg_ms_pair"],
                        "avg_ms_pair_wall": meta_payload["avg_ms_pair"] + 1.0,
                        "meta_json": str(meta_path),
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(final, "_run_subprocess", lambda cmd, cwd: fake_run(cmd, cwd=str(cwd)))
    monkeypatch.setattr(final, "_git_info", lambda repo_root: {"commit": "test", "branch": "main", "is_dirty": False})
    return calls


def test_pair_audit_valid_positive_and_negative_pairs_pass(tmp_path: Path) -> None:
    summary = _audit_rows(
        tmp_path,
        [
            _existing_pair(tmp_path, "pos", 1),
            _existing_pair(tmp_path, "neg", 0),
        ],
    )

    assert summary["pass"] is True
    assert summary["positive_count"] == 1
    assert summary["negative_count"] == 1
    assert summary["invalid_positive_count"] == 0
    assert summary["invalid_negative_count"] == 0
    assert summary["positive_count_by_finger_position"] == {"3": 1}
    assert summary["negative_count_by_finger_position"] == {"3": 1}


def test_pair_audit_positive_with_different_subjects_fails(tmp_path: Path) -> None:
    summary = _audit_rows(
        tmp_path,
        [
            _existing_pair(tmp_path, "pos", 1, subject_a="alice", subject_b="bob"),
            _existing_pair(tmp_path, "neg", 0),
        ],
    )

    assert summary["pass"] is False
    assert summary["invalid_positive_count"] == 1
    assert summary["subject_mismatch_among_positives"] == 1


def test_pair_audit_negative_with_same_subject_fails(tmp_path: Path) -> None:
    summary = _audit_rows(
        tmp_path,
        [
            _existing_pair(tmp_path, "pos", 1),
            _existing_pair(tmp_path, "neg", 0, subject_a="alice", subject_b="alice"),
        ],
    )

    assert summary["pass"] is False
    assert summary["invalid_negative_count"] == 1
    assert summary["same_subject_negatives"] == 1


def test_pair_audit_finger_mismatch_fails(tmp_path: Path) -> None:
    bad = _existing_pair(tmp_path, "pos", 1)
    bad.pop("frgp")
    bad["frgp_a"] = "3"
    bad["frgp_b"] = "4"
    summary = _audit_rows(
        tmp_path,
        [
            bad,
            _existing_pair(tmp_path, "neg", 0),
        ],
    )

    assert summary["pass"] is False
    assert summary["finger_mismatch_count"] == 1
    assert summary["invalid_positive_count"] == 1


def test_pair_audit_modality_mismatch_fails(tmp_path: Path) -> None:
    summary = _audit_rows(
        tmp_path,
        [
            _existing_pair(tmp_path, "pos", 1, path_order="plain_plain"),
            _existing_pair(tmp_path, "neg", 0),
        ],
    )

    assert summary["pass"] is False
    assert summary["modality_mismatch_count"] == 1
    assert summary["invalid_positive_count"] == 1


def test_pair_audit_duplicate_unordered_pair_fails(tmp_path: Path) -> None:
    first = _existing_pair(tmp_path, "pos", 1)
    duplicate = {**first, "pair_id": "pos-duplicate", "path_a": first["path_b"], "path_b": first["path_a"]}
    summary = _audit_rows(
        tmp_path,
        [
            first,
            duplicate,
            _existing_pair(tmp_path, "neg", 0),
        ],
    )

    assert summary["pass"] is False
    assert summary["duplicate_pair_count"] == 1
    assert summary["invalid_positive_count"] == 1


def test_deterministic_selected_pair_generation_filters_protocol(tmp_path: Path) -> None:
    rows = [_pair(f"p{idx:02d}", 1 if idx % 2 else 0, "val") for idx in range(40)]
    rows.extend(
        [
            {**_pair("bad-label", 2, "val"), "label": 2},
            {**_pair("bad-pos-subject", 1, "val"), "subject_b": "different"},
            {**_pair("bad-neg-subject", 0, "val"), "subject_b": "subject-bad-neg-subject"},
            {**_pair("bad-capture", 1, "val"), "path_b": "C:/fingerprints/val/plain_again.png"},
        ]
    )
    _write_dataset(tmp_path, "toy", {"val": rows})

    sampled_a, status_a = final.load_plain_roll_pairs(
        "toy",
        "val",
        repo_root=tmp_path,
        limit=10,
        sample_strategy="balanced_spread",
        sample_seed=13,
    )
    sampled_b, _ = final.load_plain_roll_pairs(
        "toy",
        "val",
        repo_root=tmp_path,
        limit=10,
        sample_strategy="balanced_spread",
        sample_seed=13,
    )
    first, _ = final.load_plain_roll_pairs(
        "toy",
        "val",
        repo_root=tmp_path,
        limit=10,
        sample_strategy="first",
        sample_seed=13,
    )

    assert sampled_a["pair_id"].tolist() == sampled_b["pair_id"].tolist()
    assert int((sampled_a["label"] == 1).sum()) == 5
    assert int((sampled_a["label"] == 0).sum()) == 5
    assert sampled_a["pair_id"].tolist() != first["pair_id"].tolist()
    assert set(sampled_a["finger_position"]) == {"3"}
    assert status_a["protocol_eligible_pairs"] == 40
    assert status_a["filtered_out_pairs"] == 4


def test_final_runner_uses_selected_pairs_and_val_threshold_for_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    val_rows = [
        _pair("val-neg-low", 0, "val"),
        _pair("val-neg-high", 0, "val"),
        _pair("val-pos-low", 1, "val"),
        _pair("val-pos-high", 1, "val"),
    ]
    test_rows = [
        _pair("test-neg-high", 0, "test"),
        _pair("test-neg-low", 0, "test"),
        _pair("test-pos-mid", 1, "test"),
        _pair("test-pos-high", 1, "test"),
    ]
    _write_dataset(tmp_path, "toy", {"val": val_rows, "test": test_rows})
    calls = _install_fake_evaluator(
        monkeypatch,
        {
            "val-neg-low": 0.2,
            "val-neg-high": 0.8,
            "val-pos-low": 0.1,
            "val-pos-high": 0.95,
            "test-neg-high": 0.9,
            "test-neg-low": 0.1,
            "test-pos-mid": 0.7,
            "test-pos-high": 0.85,
        },
    )

    paths = final.run_benchmark(
        datasets=("toy",),
        methods=("sift",),
        splits=("val", "test"),
        outdir=tmp_path / "out",
        target_fars=(0.0, 0.5),
        limit_per_split=0,
        repo_root=tmp_path,
    )

    thresholds = pd.read_csv(paths["thresholds"])
    zero_far = thresholds[thresholds["target_far"] == 0.0].iloc[0]
    half_far = thresholds[thresholds["target_far"] == 0.5].iloc[0]
    assert zero_far["threshold"] > 0.8
    assert zero_far["threshold"] < 0.81
    assert half_far["threshold"] == pytest.approx(0.8)
    assert half_far["calibration_false_accepts"] == 1
    assert half_far["calibration_far"] == pytest.approx(0.5)

    metrics = pd.read_csv(paths["metrics"])
    test_metric = metrics[(metrics["split"] == "test") & (metrics["target_far"] == 0.5)].iloc[0]
    assert test_metric["threshold"] == pytest.approx(0.8)
    assert test_metric["tar"] == pytest.approx(0.5)
    assert test_metric["far"] == pytest.approx(0.5)
    assert int(test_metric["ta"]) == 1
    assert int(test_metric["fr"]) == 1
    assert int(test_metric["fa"]) == 1
    assert int(test_metric["tr"]) == 1

    positive_only = pd.read_csv(paths["positive_only_metrics"])
    negative_only = pd.read_csv(paths["negative_only_metrics"])
    for _, row in positive_only.iterrows():
        assert int(row["true_accepts"]) + int(row["false_rejects"]) == int(row["n_positive"])
        assert row["tar"] == pytest.approx(row["true_accepts"] / row["n_positive"])
    for _, row in negative_only.iterrows():
        assert int(row["false_accepts"]) + int(row["true_rejects"]) == int(row["n_negative"])
        assert row["far"] == pytest.approx(row["false_accepts"] / row["n_negative"])

    pair_flags = [cmd[cmd.index("--pairs_file") + 1] for cmd in calls]
    assert pair_flags
    assert all(Path(path).parent.name == "selected_pairs" for path in pair_flags)
    assert all("data" not in Path(path).parts for path in pair_flags)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == final.OUTPUT_SCHEMA_VERSION
    assert manifest["sample_strategy"] == "balanced_spread"
    assert manifest["sample_seed"] == 13
    assert manifest["output_schema"] == final.output_schema()
    assert paths["metrics"].name == "plain_roll_final_metrics.csv"
    assert paths["positive_only_metrics"].name == "plain_roll_final_positive_only_metrics.csv"
    assert paths["negative_only_metrics"].name == "plain_roll_final_negative_only_metrics.csv"
    assert paths["thresholds"].name == "plain_roll_final_thresholds.csv"
    markdown_path = tmp_path / "out" / "final_markdown" / "toy_sift_plain_roll_final.md"
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Positive-only verification evidence" in markdown
    assert "Negative-only impostor evidence" in markdown
    summary = paths["summary"].read_text(encoding="utf-8")
    assert "TAR/FRR are computed only from positive pairs" in summary
    assert "FAR/TNR are computed only from negative pairs" in summary

    score_csv = pd.read_csv(tmp_path / "out" / "scores_toy_sift_val.csv")
    assert "pair_id" in score_csv.columns
    assert "finger_position" in score_csv.columns
    assert score_csv["pair_id"].tolist() == [row["pair_id"] for row in val_rows]
    assert score_csv["finger_position"].astype(str).tolist() == ["3", "3", "3", "3"]


def test_latency_columns_from_classic_and_minutiae_are_surfaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_pair("neg", 0, "val"), _pair("pos", 1, "val")]
    _write_dataset(tmp_path, "toy", {"val": rows})
    _install_fake_evaluator(monkeypatch, {"neg": 0.2, "pos": 0.9})

    paths = final.run_benchmark(
        datasets=("toy",),
        methods=("classic_v2", "minutiae"),
        splits=("val",),
        outdir=tmp_path / "out",
        target_fars=(0.5,),
        limit_per_split=0,
        repo_root=tmp_path,
    )

    latency = pd.read_csv(paths["latency_summary"])
    classic = latency[latency["method"] == "classic_v2"].iloc[0]
    minutiae = latency[latency["method"] == "minutiae"].iloc[0]
    assert classic["avg_ms_pair_score_csv"] == pytest.approx(6.0)
    assert classic["p50_ms_pair_score_csv"] == pytest.approx(6.0)
    assert int(classic["cache_hits"]) == 2
    assert int(classic["cache_misses"]) == 3
    assert classic["method_meta_json"].endswith("scores_toy_classic_v2_val.csv.meta.json")
    assert minutiae["avg_ms_pair_score_csv"] == pytest.approx(9.0)
    assert minutiae["meta_avg_ms_pair"] == pytest.approx(9.0)
    assert int(minutiae["cache_hits"]) == 4
    assert int(minutiae["cache_misses"]) == 5


def test_select_pairs_only_writes_audits_without_invoking_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dataset(
        tmp_path,
        "toy",
        {
            "val": [
                _existing_pair(tmp_path, "pos", 1),
                _existing_pair(tmp_path, "neg", 0),
            ]
        },
    )
    calls = _install_fake_evaluator(monkeypatch, {"pos": 0.9, "neg": 0.1})

    paths = final.run_benchmark(
        datasets=("toy",),
        methods=("sift",),
        splits=("val",),
        outdir=tmp_path / "out",
        target_fars=(0.5,),
        limit_per_split=0,
        repo_root=tmp_path,
        select_pairs_only=True,
        strict_pair_audit=True,
    )

    assert calls == []
    assert paths["selected_pairs_toy_val"].exists()
    assert paths["pair_audit_json_toy_val"].exists()
    assert paths["pair_audit_markdown_toy_val"].exists()
    assert paths["pair_audit_summary"].exists()
    assert "metrics" not in paths
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["select_pairs_only"] is True
    assert manifest["pair_audits"][0]["pass"] is True


def test_strict_audit_failure_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_dataset(
        tmp_path,
        "toy",
        {
            "val": [
                _pair("pos-missing", 1, "val"),
                _pair("neg-missing", 0, "val"),
            ]
        },
    )
    monkeypatch.setattr(final, "REPO_ROOT", tmp_path)

    code = final.main(
        [
            "--datasets",
            "toy",
            "--methods",
            "sift",
            "--splits",
            "val",
            "--outdir",
            str(tmp_path / "out"),
            "--target_far",
            "0.5",
            "--limit_per_split",
            "0",
            "--select_pairs_only",
            "--strict_pair_audit",
        ]
    )

    assert code == 2
    assert (tmp_path / "out" / "pair_audit" / "pair_audit_toy_val.json").exists()


def test_sourceafis_runner_contract_is_unchanged() -> None:
    assert sourceafis.OUTPUT_SCHEMA_VERSION == "sourceafis_open_plain_roll_benchmark_v1"
    assert sourceafis.DEFAULT_SAMPLE_STRATEGY == "balanced_spread"
    assert "sourceafis_plain_roll_scores_val.csv" in sourceafis.output_schema()
    assert "method" not in sourceafis.SCORES_COLUMNS
    assert not any("sourceafis" in method for method in final.DEFAULT_METHODS)
