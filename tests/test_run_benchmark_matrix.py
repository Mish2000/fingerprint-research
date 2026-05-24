import csv
from pathlib import Path

import pytest

from pipelines.benchmark import run_benchmark_matrix as matrix
from pipelines.benchmark.method_profiles import load_benchmark_method_profiles


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


def test_build_eval_cmd_supports_minutiae(tmp_path: Path) -> None:
    cmd = matrix.build_eval_cmd(
        outdir=tmp_path,
        dataset="demo_ds",
        data_dir=tmp_path / "dataset",
        method="minutiae",
        split="test",
        limit=25,
        ensure_pairs=False,
        dedicated_ckpt="auto",
    )

    assert cmd[cmd.index("--method") + 1] == "minutiae"
    assert cmd[cmd.index("--minutiae_target_size") + 1] == "512"
    assert cmd[cmd.index("--minutiae_spatial_tolerance") + 1] == "14.0"
    assert cmd[cmd.index("--minutiae_min_required_minutiae") + 1] == "12"


@pytest.mark.parametrize(
    ("method", "expected_backbone"),
    [
        ("dl_quick", "resnet18"),
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


def test_default_methods_are_canonical_only() -> None:
    methods = matrix.resolve_methods_request(
        methods_raw="",
        profile="canonical",
        include_research_methods=False,
    )

    assert methods == ["classic_v2", "minutiae", "harris", "sift", "dl_quick", "vit"]
    assert "dedicated" not in methods


def test_benchmark_method_profiles_are_loaded_from_registry() -> None:
    profiles = load_benchmark_method_profiles()

    assert profiles.loaded_from_registry is True
    assert list(profiles.canonical) == ["classic_v2", "minutiae", "harris", "sift", "dl_quick", "vit"]
    assert "dedicated" not in profiles.canonical
    assert "sift_plain_roll_v2" not in profiles.canonical
    assert list(profiles.research_methods) == ["sift_plain_roll_v2", "dedicated"]
    assert list(profiles.dedicated) == ["dedicated"]
    assert matrix.BENCHMARK_METHOD_PROFILES == {
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


def test_research_profile_includes_dedicated_explicitly() -> None:
    methods = matrix.resolve_methods_request(
        methods_raw="",
        profile="research",
        include_research_methods=False,
    )

    assert methods == [
        "classic_v2",
        "minutiae",
        "harris",
        "sift",
        "dl_quick",
        "vit",
        "sift_plain_roll_v2",
        "dedicated",
    ]


def test_explicit_methods_can_run_dedicated_only() -> None:
    methods = matrix.resolve_methods_request(
        methods_raw="dedicated",
        profile="canonical",
        include_research_methods=False,
    )

    assert methods == ["dedicated"]


def test_include_research_methods_appends_dedicated() -> None:
    methods = matrix.resolve_methods_request(
        methods_raw="sift,vit",
        profile="canonical",
        include_research_methods=True,
    )

    assert methods == ["sift", "vit", "sift_plain_roll_v2", "dedicated"]


def test_build_eval_cmd_supports_sift_plain_roll_v2_research_method(tmp_path: Path) -> None:
    cmd = matrix.build_eval_cmd(
        outdir=tmp_path,
        dataset="demo_ds",
        data_dir=tmp_path / "dataset",
        method="sift_plain_roll_v2",
        split="test",
        limit=25,
        ensure_pairs=False,
        dedicated_ckpt="auto",
    )

    assert cmd[cmd.index("--method") + 1] == "sift_plain_roll_v2"
    assert cmd[cmd.index("--detector") + 1] == "sift"
    assert cmd[cmd.index("--score_mode") + 1] == "inliers_times_inlier_ratio_times_log1p_matches"
    assert cmd[cmd.index("--nfeatures") + 1] == "3000"
    assert cmd[cmd.index("--target_size") + 1] == "768"
    assert cmd[cmd.index("--blur_ksize") + 1] == "0"
    assert cmd[cmd.index("--ransac_model") + 1] == "affine_full_2d"
    assert cmd[cmd.index("--ransac_thresh") + 1] == "3.0"


def test_official_run_path_preserves_sift_v2_and_canonical_sift_provenance_defaults(tmp_path: Path) -> None:
    def value(cmd: list[str], flag: str) -> str:
        return cmd[cmd.index(flag) + 1]

    common = {
        "outdir": tmp_path,
        "dataset": "demo_ds",
        "data_dir": tmp_path / "dataset",
        "split": "test",
        "limit": 0,
        "ensure_pairs": False,
        "dedicated_ckpt": "auto",
    }
    v2_cmd = matrix.build_eval_cmd(method="sift_plain_roll_v2", **common)
    sift_cmd = matrix.build_eval_cmd(method="sift", **common)

    assert value(v2_cmd, "--method") == "sift_plain_roll_v2"
    assert value(v2_cmd, "--detector") == "sift"
    assert value(v2_cmd, "--target_size") == "768"
    assert value(v2_cmd, "--nfeatures") == "3000"
    assert value(v2_cmd, "--blur_ksize") == "0"
    assert value(v2_cmd, "--ratio") == "0.75"
    assert value(v2_cmd, "--ransac_model") == "affine_full_2d"
    assert value(v2_cmd, "--ransac_thresh") == "3.0"
    assert value(v2_cmd, "--score_mode") == "inliers_times_inlier_ratio_times_log1p_matches"

    assert value(sift_cmd, "--method") == "sift"
    assert value(sift_cmd, "--detector") == "sift"
    assert value(sift_cmd, "--target_size") == "512"
    assert value(sift_cmd, "--nfeatures") == "1500"
    assert value(sift_cmd, "--blur_ksize") == "3"
    assert value(sift_cmd, "--ratio") == "0.75"
    assert value(sift_cmd, "--ransac_model") == "homography"
    assert value(sift_cmd, "--ransac_thresh") == "3.0"
    assert value(sift_cmd, "--score_mode") == "inliers_over_min_keypoints"


def test_render_results_md_orders_fusion_after_source_methods(tmp_path: Path) -> None:
    summary_csv = tmp_path / "results_summary.csv"
    summary_md = tmp_path / "results_summary.md"

    cols = [
        "method",
        "split",
        "n_pairs",
        "auc",
        "eer",
        "eer_threshold",
        "far_at_eer",
        "frr_at_eer",
        "tar_at_far_1e_2",
        "frr_at_far_1e_2",
        "tar_at_far_1e_3",
        "frr_at_far_1e_3",
        "avg_ms_pair_reported",
        "avg_ms_pair_wall",
    ]
    rows = [
        {
            "method": matrix.FUSION_METHOD,
            "split": "val",
            "n_pairs": 1,
            "auc": 0.7,
            "eer": 0.2,
            "eer_threshold": 0.55,
            "far_at_eer": 0.2,
            "frr_at_eer": 0.2,
            "tar_at_far_1e_2": 0.5,
            "frr_at_far_1e_2": 0.5,
            "tar_at_far_1e_3": 0.4,
            "frr_at_far_1e_3": 0.6,
            "avg_ms_pair_reported": 1.0,
            "avg_ms_pair_wall": 2.0,
        },
        {
            "method": "vit",
            "split": "val",
            "n_pairs": 1,
            "auc": 0.7,
            "eer": 0.2,
            "eer_threshold": 0.55,
            "far_at_eer": 0.2,
            "frr_at_eer": 0.2,
            "tar_at_far_1e_2": 0.5,
            "frr_at_far_1e_2": 0.5,
            "tar_at_far_1e_3": 0.4,
            "frr_at_far_1e_3": 0.6,
            "avg_ms_pair_reported": 1.0,
            "avg_ms_pair_wall": 2.0,
        },
        {
            "method": "dl_quick",
            "split": "val",
            "n_pairs": 1,
            "auc": 0.7,
            "eer": 0.2,
            "eer_threshold": 0.55,
            "far_at_eer": 0.2,
            "frr_at_eer": 0.2,
            "tar_at_far_1e_2": 0.5,
            "frr_at_far_1e_2": 0.5,
            "tar_at_far_1e_3": 0.4,
            "frr_at_far_1e_3": 0.6,
            "avg_ms_pair_reported": 1.0,
            "avg_ms_pair_wall": 2.0,
        },
        {
            "method": "sift",
            "split": "val",
            "n_pairs": 1,
            "auc": 0.7,
            "eer": 0.2,
            "eer_threshold": 0.55,
            "far_at_eer": 0.2,
            "frr_at_eer": 0.2,
            "tar_at_far_1e_2": 0.5,
            "frr_at_far_1e_2": 0.5,
            "tar_at_far_1e_3": 0.4,
            "frr_at_far_1e_3": 0.6,
            "avg_ms_pair_reported": 1.0,
            "avg_ms_pair_wall": 2.0,
        },
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    matrix.render_results_md(summary_csv, summary_md)

    lines = [line for line in summary_md.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    assert "eer_threshold" in lines[0]
    assert "far_at_eer" in lines[0]
    assert "frr_at_eer" in lines[0]
    assert "frr_at_far_1e_2" in lines[0]
    assert "frr_at_far_1e_3" in lines[0]
    body_lines = lines[2:]
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
    assert args.methods == ""
    assert args.profile == "canonical"
    assert args.include_research_methods is False
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
