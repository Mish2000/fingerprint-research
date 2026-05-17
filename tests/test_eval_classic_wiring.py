import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.benchmark import eval_classic
from pipelines.benchmark import eval_quick
from pipelines.benchmark import evaluate


def _write_dataset_dir(root: Path) -> Path:
    data_dir = root / "dataset"
    data_dir.mkdir()
    (data_dir / "manifest.csv").write_text("pair_id\n1\n", encoding="utf-8")
    (data_dir / "pairs_val.csv").write_text(
        "path_a,path_b,label\nimg_a.png,img_b.png,1\nimg_c.png,img_d.png,0\n",
        encoding="utf-8",
    )
    return data_dir


def _metrics(
    auc: float = 0.99,
    eer: float = 0.01,
    eer_threshold: float = 0.5,
    far_at_eer: float = 0.01,
    frr_at_eer: float = 0.01,
) -> evaluate.AucEerMetrics:
    return evaluate.AucEerMetrics(
        auc=auc,
        eer=eer,
        eer_threshold=eer_threshold,
        far_at_eer=far_at_eer,
        frr_at_eer=frr_at_eer,
    )


def test_compute_auc_eer_exposes_far_frr_details() -> None:
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.7, 0.4, 0.2])

    metrics = evaluate.compute_auc_eer(y_true, scores)

    assert metrics.auc == pytest.approx(1.0)
    assert metrics.eer == pytest.approx(0.0)
    assert np.isfinite(metrics.far_at_eer)
    assert np.isfinite(metrics.frr_at_eer)


def test_append_summary_row_writes_far_frr_columns(tmp_path: Path) -> None:
    summary_csv = tmp_path / "results_summary.csv"
    row = evaluate.EvalRow(
        timestamp_utc="2026-05-09T00:00:00Z",
        method="dl_quick",
        split="val",
        n_pairs=2,
        auc=1.0,
        eer=0.0,
        eer_threshold=0.7,
        far_at_eer=0.0,
        frr_at_eer=0.0,
        tar_at_far_1e_2=1.0,
        frr_at_far_1e_2=0.0,
        tar_at_far_1e_3=1.0,
        frr_at_far_1e_3=0.0,
        avg_ms_pair_reported=None,
        avg_ms_pair_wall=1.0,
        scores_csv="scores.csv",
        meta_json=None,
        config_json="{}",
    )

    evaluate.append_summary_row(summary_csv, row)

    df = pd.read_csv(summary_csv)
    assert {
        "eer_threshold",
        "far_at_eer",
        "frr_at_eer",
        "frr_at_far_1e_2",
        "frr_at_far_1e_3",
    }.issubset(df.columns)


def test_append_summary_row_upgrades_existing_summary_header(tmp_path: Path) -> None:
    summary_csv = tmp_path / "results_summary.csv"
    old_columns = [
        "timestamp_utc",
        "method",
        "split",
        "n_pairs",
        "auc",
        "eer",
        "tar_at_far_1e_2",
        "tar_at_far_1e_3",
        "avg_ms_pair_reported",
        "avg_ms_pair_wall",
        "scores_csv",
        "meta_json",
        "config_json",
    ]
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-05-08T00:00:00Z",
                "method": "classic_v2",
                "split": "val",
                "n_pairs": 2,
                "auc": 0.5,
                "eer": 0.5,
                "tar_at_far_1e_2": 0.1,
                "tar_at_far_1e_3": 0.0,
                "avg_ms_pair_reported": "",
                "avg_ms_pair_wall": 1.0,
                "scores_csv": "old_scores.csv",
                "meta_json": "",
                "config_json": "{}",
            }
        ],
        columns=old_columns,
    ).to_csv(summary_csv, index=False)
    row = evaluate.EvalRow(
        timestamp_utc="2026-05-09T00:00:00Z",
        method="dl_quick",
        split="val",
        n_pairs=2,
        auc=1.0,
        eer=0.0,
        eer_threshold=0.7,
        far_at_eer=0.0,
        frr_at_eer=0.0,
        tar_at_far_1e_2=1.0,
        frr_at_far_1e_2=0.0,
        tar_at_far_1e_3=1.0,
        frr_at_far_1e_3=0.0,
        avg_ms_pair_reported=None,
        avg_ms_pair_wall=1.0,
        scores_csv="scores.csv",
        meta_json=None,
        config_json="{}",
    )

    evaluate.append_summary_row(summary_csv, row)

    df = pd.read_csv(summary_csv)
    assert len(df) == 2
    assert list(df.columns[: len(evaluate.SUMMARY_HEADER)]) == evaluate.SUMMARY_HEADER
    assert df.loc[1, "far_at_eer"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("method", "expected_detector", "expected_score_mode", "expected_ransac_thresh", "expected_semantics_epoch"),
    [
        ("classic_v2", "gftt_orb", "inliers_over_k", "4.0", None),
        ("harris", "harris_orb", "inliers_over_min_keypoints", "3.0", "harris_runtime_aligned_v1"),
        ("sift", "sift", "inliers_over_min_keypoints", "3.0", "sift_runtime_aligned_v1"),
    ],
)
def test_evaluate_classic_branch_uses_runtime_truthful_forwarding(
    monkeypatch,
    tmp_path: Path,
    method: str,
    expected_detector: str,
    expected_score_mode: str,
    expected_ransac_thresh: str,
    expected_semantics_epoch: str | None,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = _write_dataset_dir(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_subprocess(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd

    def fake_save_roc_png(_y, _s, out_png: Path, title: str) -> None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"PNG")

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(evaluate, "run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(evaluate, "read_scores", lambda _path: (np.array([1, 0]), np.array([0.9, 0.1])))
    monkeypatch.setattr(evaluate, "compute_auc_eer", lambda _y, _s: _metrics())
    monkeypatch.setattr(evaluate, "tar_at_far", lambda _y, _s, _far: 0.95)
    monkeypatch.setattr(evaluate, "save_roc_png", fake_save_roc_png)
    monkeypatch.setattr(
        evaluate,
        "append_summary_row",
        lambda _summary_csv, row: captured.setdefault("row", row),
    )

    summary_csv = tmp_path / "results_summary.csv"
    out_scores = tmp_path / "scores.csv"
    out_roc = tmp_path / "roc.png"
    out_run_meta = tmp_path / "run.meta.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--method",
            method,
            "--dataset",
            "demo_ds",
            "--data_dir",
            str(data_dir),
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

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[1] == str(repo_root / "pipelines" / "benchmark" / "eval_classic.py")
    assert cmd[cmd.index("--detector") + 1] == expected_detector
    assert cmd[cmd.index("--score_mode") + 1] == expected_score_mode
    assert cmd[cmd.index("--ransac_thresh") + 1] == expected_ransac_thresh
    assert cmd[cmd.index("--pairs") + 1] == str(data_dir / "pairs_val.csv")
    if method in {"harris", "sift"}:
        assert cmd[cmd.index("--target_size") + 1] == "512"

    row = captured["row"]
    config = json.loads(row.config_json)
    assert config["classic"]["detector"] == expected_detector
    assert config["classic"]["score_mode"] == expected_score_mode
    assert str(config["classic"]["ransac_thresh"]) == expected_ransac_thresh
    assert config["method_semantics_epoch"] == expected_semantics_epoch
    assert out_run_meta.exists()


@pytest.mark.parametrize(
    ("method", "expected_backbone"),
    [
        ("dl_quick", "resnet18"),
        ("vit", "vit_base"),
    ],
)
def test_evaluate_dl_branch_forwards_explicit_no_mask_ablation(
    monkeypatch,
    tmp_path: Path,
    method: str,
    expected_backbone: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = _write_dataset_dir(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_subprocess(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd

    def fake_save_roc_png(_y, _s, out_png: Path, title: str) -> None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"PNG")

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(evaluate, "run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(evaluate, "read_scores", lambda _path: (np.array([1, 0]), np.array([0.9, 0.1])))
    monkeypatch.setattr(evaluate, "compute_auc_eer", lambda _y, _s: _metrics())
    monkeypatch.setattr(evaluate, "tar_at_far", lambda _y, _s, _far: 0.95)
    monkeypatch.setattr(evaluate, "save_roc_png", fake_save_roc_png)
    monkeypatch.setattr(evaluate, "append_summary_row", lambda _summary_csv, _row: None)

    summary_csv = tmp_path / "results_summary.csv"
    out_scores = tmp_path / "scores.csv"
    out_roc = tmp_path / "roc.png"
    out_run_meta = tmp_path / "run.meta.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--method",
            method,
            "--dataset",
            "demo_ds",
            "--data_dir",
            str(data_dir),
            "--summary_csv",
            str(summary_csv),
            "--out_scores",
            str(out_scores),
            "--out_roc",
            str(out_roc),
            "--out_run_meta",
            str(out_run_meta),
            "--no_mask",
        ],
    )

    evaluate.main()

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[1] == str(repo_root / "pipelines" / "benchmark" / "eval_quick.py")
    assert cmd[cmd.index("--backbone") + 1] == expected_backbone
    assert "--no_mask" in cmd
    assert out_run_meta.exists()


def test_evaluate_minutiae_branch_forwards_dedicated_script_and_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = _write_dataset_dir(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_subprocess(cmd, *, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd

    def fake_save_roc_png(_y, _s, out_png: Path, title: str) -> None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.write_bytes(b"PNG")

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(evaluate, "run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(evaluate, "read_scores", lambda _path: (np.array([1, 0]), np.array([0.9, 0.1])))
    monkeypatch.setattr(evaluate, "compute_auc_eer", lambda _y, _s: _metrics())
    monkeypatch.setattr(evaluate, "tar_at_far", lambda _y, _s, _far: 0.95)
    monkeypatch.setattr(evaluate, "save_roc_png", fake_save_roc_png)
    monkeypatch.setattr(
        evaluate,
        "append_summary_row",
        lambda _summary_csv, row: captured.setdefault("row", row),
    )

    summary_csv = tmp_path / "results_summary.csv"
    out_scores = tmp_path / "scores.csv"
    out_roc = tmp_path / "roc.png"
    out_run_meta = tmp_path / "run.meta.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--method",
            "minutiae",
            "--dataset",
            "demo_ds",
            "--data_dir",
            str(data_dir),
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

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[1] == str(repo_root / "pipelines" / "benchmark" / "eval_minutiae.py")
    assert cmd[cmd.index("--pairs") + 1] == str(data_dir / "pairs_val.csv")
    assert cmd[cmd.index("--target_size") + 1] == "512"
    assert cmd[cmd.index("--spatial_tolerance") + 1] == "14.0"

    row = captured["row"]
    config = json.loads(row.config_json)
    assert config["method"] == "minutiae"
    assert config["method_semantics_epoch"] == "minutiae_crossing_number_aligned_v2"
    assert config["minutiae"]["crossing_number"] == {"ridge_ending": 1, "bifurcation": 3}
    assert config["minutiae"]["alignment"] == "anchor_orientation_rotation_translation"
    assert out_run_meta.exists()


def test_eval_quick_no_mask_flag_disables_masking_in_model_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data_dir = _write_dataset_dir(tmp_path)
    out_csv = tmp_path / "scores.csv"
    captured: dict[str, object] = {}

    class FakeBaselineDL:
        def __init__(self, dl_cfg, prep_cfg, device=None):
            captured["use_mask"] = dl_cfg.use_mask
            captured["backbone"] = dl_cfg.backbone
            self.embed_dim = {"resnet18": 512, "resnet50": 2048, "vit_base": 768}[dl_cfg.backbone]
            self._cfg = {
                "dl_cfg": asdict(dl_cfg),
                "prep_cfg": asdict(prep_cfg),
                "embed_dim": self.embed_dim,
                "expected_embed_dim": self.embed_dim,
                "pretrained_required": True,
                "pretrained_loaded": True,
            }
            self.device = "cpu"

        def config_dict(self):
            return dict(self._cfg)

        def embed_path(self, path: str, capture=None):
            name = Path(path).name
            vec = np.zeros(self.embed_dim, dtype=np.float32)
            if name in {"img_a.png", "img_b.png", "img_c.png"}:
                vec[0] = 1.0
                return vec, 1.0
            vec[0] = -1.0
            return vec, 1.0

        def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denom)

    monkeypatch.setattr(eval_quick, "BaselineDL", FakeBaselineDL)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_quick.py",
            str(out_csv),
            "--pairs",
            str(data_dir / "pairs_val.csv"),
            "--split",
            "val",
            "--dataset",
            "demo_ds",
            "--backbone",
            "resnet50",
            "--no_mask",
        ],
    )

    eval_quick.main()

    assert captured["backbone"] == "resnet50"
    assert captured["use_mask"] is False
    assert out_csv.exists()
    assert '"use_mask": false' in out_csv.with_suffix(".meta.json").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("detector", ["orb", "gftt_orb", "harris_orb", "sift"])
def test_eval_classic_main_writes_actual_extracted_keypoint_counts(
    monkeypatch,
    tmp_path: Path,
    detector: str,
) -> None:
    data_dir = _write_dataset_dir(tmp_path)
    out_csv = tmp_path / "scores.csv"
    keypoint_counts = {
        "img_a.png": 11,
        "img_b.png": 7,
        "img_c.png": 5,
        "img_d.png": 3,
    }

    def fake_extract(path_str: str, detector_name: str, nfeatures: int, long_edge: int, target_size: int):
        del detector_name, nfeatures, long_edge, target_size
        count = keypoint_counts[Path(path_str).name]
        return [object()] * count, np.ones((count, 1), dtype=np.uint8), None

    def fake_match_and_score(
        des1,
        des2,
        kps1,
        kps2,
        score_mode: str,
        ratio: float,
        ransac_thresh: float,
        detector: str,
        normalization_k: int,
    ) -> tuple[float, int, int]:
        del des1, des2, kps1, kps2, score_mode, ratio, ransac_thresh, detector, normalization_k
        return 0.42, 6, 9

    monkeypatch.setattr(eval_classic, "extract", fake_extract)
    monkeypatch.setattr(eval_classic, "match_and_score", fake_match_and_score)
    monkeypatch.setattr(eval_classic, "compute_auc_eer", lambda _y, _s: (0.5, 0.5))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_classic.py",
            str(out_csv),
            "--pairs",
            str(data_dir / "pairs_val.csv"),
            "--split",
            "val",
            "--detector",
            detector,
            "--nfeatures",
            "1500",
        ],
    )

    eval_classic.main()

    df = pd.read_csv(out_csv)

    assert list(df.columns) == ["label", "split", "path_a", "path_b", "score", "inliers", "matches", "k1", "k2"]
    assert df["k1"].tolist() == [11, 5]
    assert df["k2"].tolist() == [7, 3]
    assert df["score"].tolist() == [0.42, 0.42]
    assert df["inliers"].tolist() == [6, 6]
    assert df["matches"].tolist() == [9, 9]


def test_week3_score_pairs_shim_routes_to_eval_classic() -> None:
    shim_path = Path(__file__).resolve().parents[1] / "research_history" / "week03" / "week3_score_pairs.py"
    spec = importlib.util.spec_from_file_location("week3_score_pairs_shim", shim_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main is eval_classic.main
