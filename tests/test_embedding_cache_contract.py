from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipelines.benchmark import check_embedding_cache, eval_quick, warm_dl_cache
from pipelines.benchmark.embedding_cache import build_cache_key_config, cache_entry_error
from src.fpbench.matchers.baseline_dl import DLBaselineConfig
from src.fpbench.preprocess.preprocess import PreprocessConfig


def test_check_warm_and_eval_use_identical_cache_key_config_and_path(tmp_path: Path) -> None:
    dl_cfg = DLBaselineConfig(backbone="resnet18", use_mask=True)
    prep_cfg = PreprocessConfig(target_size=512)
    cfg_for_key = build_cache_key_config(dl_cfg, prep_cfg)
    cfg_json = json.dumps(cfg_for_key, sort_keys=True, ensure_ascii=False)
    source_path = r"C:\fingerprint collected data\subject_001\plain.png"
    cache_root = tmp_path / "cache"

    assert cfg_for_key == {
        "dl_cfg": {
            "backbone": "resnet18",
            "input_size": 224,
            "use_mask": True,
            "roi_min_frac": 0.02,
            "roi_max_frac": 0.85,
            "gate_top_plain": 0.18,
            "gate_top_roll": 0.05,
            "gate_border": 12,
        },
        "prep_cfg": {
            "target_size": 512,
            "clahe_clip": 2.0,
            "clahe_grid": (8, 8),
            "blur_ksize": 3,
        },
        "embed_dim": 512,
        "expected_embed_dim": 512,
        "pretrained_required": True,
        "pretrained_loaded": True,
    }

    assert check_embedding_cache.cache_file_for(cache_root, source_path, cfg_json, "") == warm_dl_cache.cache_file_for(
        cache_root,
        source_path,
        cfg_json,
        "",
    )
    assert check_embedding_cache.cache_file_for(cache_root, source_path, cfg_json, "") == eval_quick.cache_file_for(
        cache_root,
        source_path,
        cfg_json,
        "",
    )


def test_shape_correct_stale_cache_without_metadata_is_invalid(tmp_path: Path) -> None:
    stale_cache = tmp_path / "stale.npz"
    np.savez_compressed(stale_cache, emb=np.ones(512, dtype=np.float32))

    assert cache_entry_error(stale_cache, backbone="resnet18", expected_dim=512) == "missing metadata field: backbone"
    assert warm_dl_cache.cache_entry_is_valid(stale_cache, backbone="resnet18", expected_dim=512) is False


def test_cache_validation_requires_pretrained_metadata_true(tmp_path: Path) -> None:
    missing_flags = tmp_path / "missing_flags.npz"
    np.savez_compressed(
        missing_flags,
        emb=np.ones(512, dtype=np.float32),
        backbone=np.array("resnet18"),
        embed_dim=np.array(512),
    )
    assert (
        cache_entry_error(missing_flags, backbone="resnet18", expected_dim=512)
        == "missing metadata field: pretrained_required"
    )

    false_loaded = tmp_path / "false_loaded.npz"
    np.savez_compressed(
        false_loaded,
        emb=np.ones(512, dtype=np.float32),
        backbone=np.array("resnet18"),
        embed_dim=np.array(512),
        pretrained_required=np.array(True),
        pretrained_loaded=np.array(False),
    )
    assert cache_entry_error(false_loaded, backbone="resnet18", expected_dim=512) == "pretrained_loaded is not true"
