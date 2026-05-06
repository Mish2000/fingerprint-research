from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pipelines.benchmark.eval_patch_descriptor_sanity import load_sanity_checkpoint
from src.fpbench.matchers.dedicated_matcher import DedicatedMatcher, SimCLRModel, _geometry_aware_score


def test_geometry_changes_ranking_even_with_same_descriptor_similarity():
    weak, weak_meta = _geometry_aware_score(0.74, 0.72, 0.10, 0.05)
    strong, strong_meta = _geometry_aware_score(0.74, 0.72, 0.90, 0.85)
    assert strong > weak
    assert "raw_score" in weak_meta
    assert "raw_score" in strong_meta


def _write_smoke_checkpoint(
    path: Path,
    *,
    model_arch: str,
    include_model_config: bool = True,
) -> Path:
    model = SimCLRModel(emb_dim=256, proj_dim=128, model_arch=model_arch)
    ckpt = {
        "step": 1,
        "model": model.state_dict(),
        "args": {
            "patch": 48,
            "max_kpts": 800,
            "patches_per_image": 8,
            "emb_dim": 256,
            "proj_dim": 128,
            "model_arch": model_arch,
        },
    }
    if include_model_config:
        ckpt["model_config"] = {
            "dedicated_model_version": 2,
            "model_arch": model_arch,
            "patch": 48,
            "emb_dim": 256,
            "proj_dim": 128,
            "encoder_channels": [32, 64, 128] if model_arch == "v1_small_cnn" else [48, 96, 160, 192],
            "train": {"steps": 1, "batch": 2, "seed": 123},
        }
    torch.save(ckpt, path)
    return path


def test_dedicated_matcher_loads_legacy_v1_checkpoint_without_model_config(tmp_path: Path) -> None:
    ckpt_path = _write_smoke_checkpoint(
        tmp_path / "legacy_v1_smoke.pth",
        model_arch="v1_small_cnn",
        include_model_config=False,
    )

    matcher = DedicatedMatcher(ckpt_path=str(ckpt_path), device="cpu")

    assert matcher.model_arch == "v1_small_cnn"
    assert matcher.emb_dim == 256
    assert matcher.model_config["legacy_assumed_config"] is True


def test_dedicated_matcher_loads_v2_checkpoint_from_model_config(tmp_path: Path) -> None:
    ckpt_path = _write_smoke_checkpoint(
        tmp_path / "v2_smoke.pth",
        model_arch="v2_medium_cnn",
    )

    matcher = DedicatedMatcher(ckpt_path=str(ckpt_path), device="cpu")

    assert matcher.model_arch == "v2_medium_cnn"
    assert matcher.emb_dim == 256
    assert matcher.proj_dim == 128
    assert matcher.model_config["encoder_channels"] == [48, 96, 160, 192]


def test_sanity_checkpoint_loader_reads_model_config(tmp_path: Path) -> None:
    ckpt_path = _write_smoke_checkpoint(
        tmp_path / "sanity_v2_smoke.pth",
        model_arch="v2_medium_cnn",
    )

    _, model_config = load_sanity_checkpoint(ckpt_path)

    assert model_config["model_arch"] == "v2_medium_cnn"
    assert model_config["patch"] == 48
    assert model_config["emb_dim"] == 256
    assert model_config["proj_dim"] == 128


def test_dedicated_matcher_rejects_unknown_model_arch(tmp_path: Path) -> None:
    ckpt_path = _write_smoke_checkpoint(
        tmp_path / "unknown_arch.pth",
        model_arch="v1_small_cnn",
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    ckpt["model_config"]["model_arch"] = "v9_mystery_cnn"
    torch.save(ckpt, ckpt_path)

    with pytest.raises(ValueError, match="Unsupported Dedicated checkpoint model_arch='v9_mystery_cnn'"):
        DedicatedMatcher(ckpt_path=str(ckpt_path), device="cpu")
