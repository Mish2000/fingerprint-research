import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from pipelines.benchmark import evaluate
from pipelines.benchmark import run_benchmark_matrix as matrix
from pipelines.benchmark import run_professor_1000_pos_neg as professor


def _pair_rows(split: str, *, n_positive: int, n_negative: int, start_pair_id: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pair_id = start_pair_id
    for idx in range(n_negative):
        rows.append(
            {
                "pair_id": pair_id,
                "label": 0,
                "split": split,
                "subject_a": 1000 + idx,
                "subject_b": 2000 + idx,
                "frgp": idx % 10 + 1,
                "path_a": f"{split}/neg/{idx}_a.png",
                "path_b": f"{split}/neg/{idx}_b.png",
            }
        )
        pair_id += 1
    for idx in range(n_positive):
        rows.append(
            {
                "pair_id": pair_id,
                "label": 1,
                "split": split,
                "subject_a": 3000 + idx,
                "subject_b": 3000 + idx,
                "frgp": idx % 10 + 1,
                "path_a": f"{split}/pos/{idx}_a.png",
                "path_b": f"{split}/pos/{idx}_b.png",
            }
        )
        pair_id += 1
    return rows


def _write_source_pair_files(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    pd.DataFrame(_pair_rows("val", n_positive=600, n_negative=600, start_pair_id=0)).to_csv(
        data_dir / "pairs_val.csv", index=False
    )
    pd.DataFrame(_pair_rows("test", n_positive=600, n_negative=600, start_pair_id=10_000)).to_csv(
        data_dir / "pairs_test.csv", index=False
    )
    pd.DataFrame(_pair_rows("train", n_positive=1200, n_negative=1200, start_pair_id=20_000)).to_csv(
        data_dir / "pairs_train.csv", index=False
    )


def test_selected_pairs_are_deterministic_exact_1000_and_val_test_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "manifest"
    _write_source_pair_files(data_dir)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    meta_a = professor.write_selected_pair_files(data_dir=data_dir, outdir=out_a, seed=20260518, n_per_label=1000)
    meta_b = professor.write_selected_pair_files(data_dir=data_dir, outdir=out_b, seed=20260518, n_per_label=1000)

    pos_a = out_a / "selected_pairs" / "positive_1000.csv"
    neg_a = out_a / "selected_pairs" / "negative_1000.csv"
    assert professor.sha256_file(pos_a) == professor.sha256_file(out_b / "selected_pairs" / "positive_1000.csv")
    assert professor.sha256_file(neg_a) == professor.sha256_file(out_b / "selected_pairs" / "negative_1000.csv")

    positive = pd.read_csv(pos_a)
    negative = pd.read_csv(neg_a)
    assert len(positive) == 1000
    assert len(negative) == 1000
    assert set(positive["label"].astype(int).unique()) == {1}
    assert set(negative["label"].astype(int).unique()) == {0}
    assert set(positive["split"].unique()).issubset({"val", "test"})
    assert set(negative["split"].unique()).issubset({"val", "test"})
    assert "train" not in set(positive["split"].unique())
    assert "train" not in set(negative["split"].unique())
    assert meta_a["source_counts"]["n_positive"] == 1200
    assert meta_a["source_counts"]["n_negative"] == 1200
    assert meta_b["source_counts"] == meta_a["source_counts"]


def test_thresholds_are_derived_from_mixed_val_scores_and_recorded(tmp_path: Path) -> None:
    source_dir = tmp_path / "reference"
    source_dir.mkdir()
    scores = pd.DataFrame(
        [{"label": 0, "score": i / 100.0} for i in range(100)]
        + [{"label": 1, "score": 0.995}, {"label": 1, "score": 0.5}]
    )
    for method in professor.METHODS:
        scores.to_csv(source_dir / f"scores_{method}_val.csv", index=False)

    rows = professor.derive_thresholds_from_val_scores(source_dir=source_dir, methods=professor.METHODS)
    assert [row["method"] for row in rows] == professor.METHODS
    assert all(row["calibration_false_accepts"] <= 1 for row in rows)
    assert all(row["calibration_far"] <= 0.01 for row in rows)
    assert all(row["threshold_source"].endswith(f"scores_{row['method']}_val.csv") for row in rows)

    recorded = professor.write_threshold_files(tmp_path / "out", rows)
    assert Path(recorded["csv_path"]).exists()
    assert Path(recorded["json_path"]).exists()
    payload = json.loads(Path(recorded["json_path"]).read_text(encoding="utf-8"))
    assert payload["target_far"] == pytest.approx(0.01)
    assert len(payload["rows"]) == len(professor.METHODS)


def _write_dataset_dir(root: Path) -> tuple[Path, Path]:
    data_dir = root / "dataset"
    data_dir.mkdir()
    (data_dir / "manifest.csv").write_text("pair_id\n1\n", encoding="utf-8")
    (data_dir / "pairs_val.csv").write_text(
        "pair_id,label,split,subject_a,subject_b,frgp,path_a,path_b\n"
        "0,1,val,1,1,1,img_a.png,img_b.png\n"
        "1,0,val,1,2,1,img_a.png,img_c.png\n",
        encoding="utf-8",
    )
    pairs_file = root / "custom_pairs.csv"
    pairs_file.write_text(
        "pair_id,label,split,subject_a,subject_b,frgp,path_a,path_b\n",
        encoding="utf-8",
    )
    return data_dir, pairs_file


def _run_fake_evaluate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pair_set_name: str,
    labels: list[int],
    scores: list[float],
) -> tuple[pd.Series, dict, dict[str, object], Path]:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir, pairs_file = _write_dataset_dir(tmp_path)
    out_scores = tmp_path / f"scores_classic_v2_{pair_set_name}.csv"
    out_roc = tmp_path / f"roc_classic_v2_{pair_set_name}.png"
    out_run_meta = tmp_path / f"run_classic_v2_{pair_set_name}.meta.json"
    summary_csv = tmp_path / "results_summary.csv"
    captured: dict[str, object] = {}

    def fake_run_subprocess(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        pd.DataFrame({"label": labels, "score": scores}).to_csv(out_scores, index=False)

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(evaluate, "run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--method",
            "classic_v2",
            "--split",
            "val",
            "--dataset",
            "demo_ds",
            "--data_dir",
            str(data_dir),
            "--pairs_file",
            str(pairs_file),
            "--pair_set_name",
            pair_set_name,
            "--operating_threshold",
            "0.5",
            "--threshold_source",
            "unit-test-threshold",
            "--summary_csv",
            str(summary_csv),
            "--out_scores",
            str(out_scores),
            "--out_roc",
            str(out_roc),
            "--out_run_meta",
            str(out_run_meta),
        ],
    )

    evaluate.main()

    row = pd.read_csv(summary_csv).iloc[0]
    meta = json.loads(out_run_meta.read_text(encoding="utf-8"))
    return row, meta, captured, out_roc


def test_evaluate_uses_pairs_file_and_skips_roc_for_single_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row, meta, captured, out_roc = _run_fake_evaluate(
        monkeypatch,
        tmp_path,
        pair_set_name="positive_1000",
        labels=[1, 1, 1, 1],
        scores=[0.6, 0.4, 0.8, 0.2],
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--pairs") + 1] == str(tmp_path / "custom_pairs.csv")
    assert row["split"] == "positive_1000"
    assert row["pair_set_name"] == "positive_1000"
    assert pd.isna(row["auc"])
    assert pd.isna(row["eer"])
    assert row["roc_status"] == "skipped"
    assert "only one label" in row["roc_skip_reason"]
    assert meta["roc"]["status"] == "skipped"
    assert not out_roc.exists()


def test_positive_only_metrics_compute_tar_frr_and_mark_far_na(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row, _, _, _ = _run_fake_evaluate(
        monkeypatch,
        tmp_path,
        pair_set_name="positive_1000",
        labels=[1, 1, 1, 1],
        scores=[0.6, 0.4, 0.8, 0.2],
    )

    assert int(row["n_positive"]) == 4
    assert int(row["n_negative"]) == 0
    assert int(row["accepted_count"]) == 2
    assert int(row["rejected_count"]) == 2
    assert row["tar"] == pytest.approx(0.5)
    assert row["frr"] == pytest.approx(0.5)
    assert pd.isna(row["far"])


def test_negative_only_metrics_compute_far_and_mark_tar_frr_na(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row, _, _, _ = _run_fake_evaluate(
        monkeypatch,
        tmp_path,
        pair_set_name="negative_1000",
        labels=[0, 0, 0, 0],
        scores=[0.6, 0.4, 0.8, 0.2],
    )

    assert int(row["n_positive"]) == 0
    assert int(row["n_negative"]) == 4
    assert int(row["false_accept_count"]) == 2
    assert int(row["true_reject_count"]) == 2
    assert row["far"] == pytest.approx(0.5)
    assert pd.isna(row["tar"])
    assert pd.isna(row["frr"])


def test_matrix_build_eval_cmd_supports_named_custom_pair_file(tmp_path: Path) -> None:
    pairs_file = tmp_path / "positive_1000.csv"
    cmd = matrix.build_eval_cmd(
        outdir=tmp_path,
        dataset="demo_ds",
        data_dir=tmp_path / "dataset",
        method="classic_v2",
        split="val",
        limit=0,
        ensure_pairs=False,
        dedicated_ckpt="auto",
        pairs_file=pairs_file,
        pair_set_name="positive_1000",
        operating_threshold=0.42,
        threshold_source="unit-test",
    )

    assert cmd[cmd.index("--pairs_file") + 1] == str(pairs_file)
    assert cmd[cmd.index("--pair_set_name") + 1] == "positive_1000"
    assert cmd[cmd.index("--operating_threshold") + 1] == "0.42"
    assert str(tmp_path / "scores_classic_v2_positive_1000.csv") in cmd
    assert str(tmp_path / "run_classic_v2_positive_1000.meta.json") in cmd
