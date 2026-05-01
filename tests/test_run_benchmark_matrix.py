import csv
from pathlib import Path

import pytest

from pipelines.benchmark import run_benchmark_matrix as matrix


def test_build_eval_cmd_supports_fusion_balanced_v1(tmp_path: Path) -> None:
    cmd = matrix.build_eval_cmd(
        outdir=tmp_path,
        dataset="demo_ds",
        data_dir=tmp_path / "dataset",
        method=matrix.FUSION_METHOD,
        split="test",
        limit=25,
        ensure_pairs=True,
        dedicated_ckpt="auto",
        fusion_fit_split="val",
        fusion_sift_weight=0.8,
        fusion_dl_weight=0.15,
        fusion_vit_weight=0.05,
    )

    assert "--fusion_source_dir" in cmd
    assert cmd[cmd.index("--fusion_source_dir") + 1] == str(tmp_path)
    assert cmd[cmd.index("--fusion_fit_split") + 1] == "val"
    assert cmd[cmd.index("--fusion_sift_weight") + 1] == "0.8"
    assert cmd[cmd.index("--fusion_dl_weight") + 1] == "0.15"
    assert cmd[cmd.index("--fusion_vit_weight") + 1] == "0.05"


@pytest.mark.parametrize(
    ("method", "expected_backbone"),
    [
        ("dl_quick", "resnet50"),
        ("vit", "vit_base"),
    ],
)
def test_build_eval_cmd_keeps_masking_enabled_for_canonical_dl_methods(
    tmp_path: Path,
    method: str,
    expected_backbone: str,
) -> None:
    cmd = matrix.build_eval_cmd(
        outdir=tmp_path,
        dataset="demo_ds",
        data_dir=tmp_path / "dataset",
        method=method,
        split="test",
        limit=25,
        ensure_pairs=False,
        dedicated_ckpt="auto",
    )

    assert cmd[cmd.index("--backbone") + 1] == expected_backbone
    assert "--no_mask" not in cmd


@pytest.mark.parametrize(
    ("method", "detector"),
    [
        ("harris", "harris_orb"),
        ("sift", "sift"),
    ],
)
def test_build_eval_cmd_aligns_harris_and_sift_with_runtime_semantics(
    tmp_path: Path,
    method: str,
    detector: str,
) -> None:
    cmd = matrix.build_eval_cmd(
        outdir=tmp_path,
        dataset="demo_ds",
        data_dir=tmp_path / "dataset",
        method=method,
        split="test",
        limit=25,
        ensure_pairs=False,
        dedicated_ckpt="auto",
    )

    assert cmd[cmd.index("--detector") + 1] == detector
    assert cmd[cmd.index("--score_mode") + 1] == "inliers_over_min_keypoints"
    assert cmd[cmd.index("--target_size") + 1] == "512"
    assert cmd[cmd.index("--ransac_thresh") + 1] == "3.0"


def test_validate_fusion_request_rejects_missing_source_methods() -> None:
    with pytest.raises(ValueError, match="Missing"):
        matrix.validate_fusion_request(
            methods=["classic_v2", matrix.FUSION_METHOD],
            splits=["val"],
            fusion_fit_split="val",
        )


def test_validate_fusion_request_rejects_missing_fit_split() -> None:
    with pytest.raises(ValueError, match="fusion_fit_split='val'"):
        matrix.validate_fusion_request(
            methods=["sift", "dl_quick", "vit", matrix.FUSION_METHOD],
            splits=["test"],
            fusion_fit_split="val",
        )


def test_validate_fusion_request_rejects_wrong_method_order() -> None:
    with pytest.raises(ValueError, match="must appear after its source methods"):
        matrix.validate_fusion_request(
            methods=["sift", matrix.FUSION_METHOD, "dl_quick", "vit"],
            splits=["val"],
            fusion_fit_split="val",
        )


def test_render_results_md_orders_fusion_after_source_methods(tmp_path: Path) -> None:
    summary_csv = tmp_path / "results_summary.csv"
    summary_md = tmp_path / "results_summary.md"

    cols = [
        "method",
        "split",
        "n_pairs",
        "auc",
        "eer",
        "tar_at_far_1e_2",
        "tar_at_far_1e_3",
        "avg_ms_pair_reported",
        "avg_ms_pair_wall",
    ]
    rows = [
        {"method": matrix.FUSION_METHOD, "split": "val", "n_pairs": 1, "auc": 0.7, "eer": 0.2, "tar_at_far_1e_2": 0.5, "tar_at_far_1e_3": 0.4, "avg_ms_pair_reported": 1.0, "avg_ms_pair_wall": 2.0},
        {"method": "vit", "split": "val", "n_pairs": 1, "auc": 0.7, "eer": 0.2, "tar_at_far_1e_2": 0.5, "tar_at_far_1e_3": 0.4, "avg_ms_pair_reported": 1.0, "avg_ms_pair_wall": 2.0},
        {"method": "dl_quick", "split": "val", "n_pairs": 1, "auc": 0.7, "eer": 0.2, "tar_at_far_1e_2": 0.5, "tar_at_far_1e_3": 0.4, "avg_ms_pair_reported": 1.0, "avg_ms_pair_wall": 2.0},
        {"method": "sift", "split": "val", "n_pairs": 1, "auc": 0.7, "eer": 0.2, "tar_at_far_1e_2": 0.5, "tar_at_far_1e_3": 0.4, "avg_ms_pair_reported": 1.0, "avg_ms_pair_wall": 2.0},
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    matrix.render_results_md(summary_csv, summary_md)

    body_lines = [line for line in summary_md.read_text(encoding="utf-8").splitlines() if line.startswith("|")][2:]
    methods = [line.split("|")[1].strip() for line in body_lines]
    assert methods == ["sift", "dl_quick", "vit", matrix.FUSION_METHOD]


def test_build_manifest_payload_keeps_required_compat_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(matrix, "safe_pkg_version", lambda name: f"{name}-version")
    monkeypatch.setattr(matrix, "get_git_info", lambda root: matrix.GitInfo("abc123", False, "main", None))

    data_dir = tmp_path / "dataset"
    outdir = tmp_path / "out"
    payload = matrix.build_manifest_payload(
        dataset="demo_ds",
        data_dir=data_dir,
        outdir=outdir,
        methods=["classic_v2"],
        splits=["val"],
        limit=10,
        ensure_pairs=True,
        emb_cache_dir="",
        cache_write=False,
        cache_strip_prefix="",
        dedicated_ckpt="auto",
        fusion_fit_split="val",
        fusion_sift_weight=0.91,
        fusion_dl_weight=0.05,
        fusion_vit_weight=0.04,
        input_hashes={"split.json": "deadbeef"},
        mode="batch",
        argv=["--dataset", "demo_ds"],
    )

    assert payload["dataset"] == {
        "name": "demo_ds",
        "resolved_data_dir": str(data_dir),
    }
    assert payload["input_hashes"] == {"split.json": "deadbeef"}
    assert payload["file_hashes_sha256"] == {"split.json": "deadbeef"}
    assert payload["packages"] == payload["package_versions"]


def test_main_uses_canonical_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_run_matrix(args) -> int:
        captured["args"] = args
        return 17

    monkeypatch.setattr(matrix, "run_matrix", fake_run_matrix)

    assert matrix.main([]) == 17
    args = captured["args"]
    assert args.outdir == "artifacts/reports/benchmark/full_nist_sd300b"
    assert args.methods == "classic_v2,harris,sift,dl_quick,dedicated,vit"
    assert matrix.FUSION_METHOD not in args.methods.split(",")
    assert args.emb_cache_dir == "artifacts/cache/embeddings"


def test_expected_output_paths_for_batch_run(tmp_path: Path) -> None:
    expected = matrix.expected_output_paths(
        tmp_path,
        methods=["classic_v2", matrix.FUSION_METHOD],
        splits=["val"],
    )

    assert expected == [
        tmp_path / "results_summary.csv",
        tmp_path / "results_summary.md",
        tmp_path / "run_manifest.json",
        tmp_path / "run.log",
        tmp_path / "scores_classic_v2_val.csv",
        tmp_path / "roc_classic_v2_val.png",
        tmp_path / "run_classic_v2_val.meta.json",
        tmp_path / f"scores_{matrix.FUSION_METHOD}_val.csv",
        tmp_path / f"roc_{matrix.FUSION_METHOD}_val.png",
        tmp_path / f"run_{matrix.FUSION_METHOD}_val.meta.json",
    ]
