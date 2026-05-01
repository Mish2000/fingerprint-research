import importlib.util
import json
import sys
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
    monkeypatch.setattr(evaluate, "compute_auc_eer", lambda _y, _s: (0.99, 0.01))
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
        ("dl_quick", "resnet50"),
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
    monkeypatch.setattr(evaluate, "compute_auc_eer", lambda _y, _s: (0.99, 0.01))
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
            self._cfg = {
                "backbone": dl_cfg.backbone,
                "use_mask": dl_cfg.use_mask,
            }
            self.device = "cpu"

        def config_dict(self):
            return dict(self._cfg)

        def embed_path(self, path: str, capture=None):
            name = Path(path).name
            if name in {"img_a.png", "img_b.png", "img_c.png"}:
                return np.array([1.0, 0.0], dtype=np.float32), 1.0
            return np.array([-1.0, 0.0], dtype=np.float32), 1.0

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
