from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.fpbench.matchers.baseline_dl import DLBaselineConfig, expected_embed_dim_for_backbone
from src.fpbench.preprocess.preprocess import PreprocessConfig


def canonical_path(path: str, strip_prefix: str = "") -> str:
    p = str(path).replace("/", "\\")
    if strip_prefix:
        pref = strip_prefix.replace("/", "\\")
        if p.lower().startswith(pref.lower()):
            p = p[len(pref):]
    else:
        marker = "fingerprint collected data\\"
        i = p.lower().find(marker)
        if i != -1:
            p = p[i + len(marker):]
    return p.lower()


def build_cache_key_config(dl_cfg: DLBaselineConfig, prep_cfg: PreprocessConfig) -> dict[str, Any]:
    embed_dim = int(expected_embed_dim_for_backbone(dl_cfg.backbone))
    return {
        "dl_cfg": asdict(dl_cfg),
        "prep_cfg": asdict(prep_cfg),
        "embed_dim": embed_dim,
        "expected_embed_dim": embed_dim,
        "pretrained_required": True,
        "pretrained_loaded": True,
    }


def cache_key_config_json(dl_cfg: DLBaselineConfig, prep_cfg: PreprocessConfig) -> str:
    return json.dumps(build_cache_key_config(dl_cfg, prep_cfg), sort_keys=True, ensure_ascii=False)


def cache_file_for(cache_root: Path, path: str, cfg_json: str, strip_prefix: str) -> Path:
    key_src = canonical_path(path, strip_prefix) + "|" + cfg_json
    digest = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    return cache_root / digest[:2] / f"{digest}.npz"


def cache_key_config_from_loaded_model(model) -> dict[str, Any]:
    cfg = model.config_dict()
    cfg.pop("device", None)
    return cfg


def assert_cache_key_config_matches_model(
    *,
    model,
    dl_cfg: DLBaselineConfig,
    prep_cfg: PreprocessConfig,
) -> dict[str, Any]:
    expected = build_cache_key_config(dl_cfg, prep_cfg)
    actual = cache_key_config_from_loaded_model(model)
    if actual != expected:
        raise RuntimeError(
            "DL embedding cache key config drifted from BaselineDL.config_dict(). "
            f"manual={expected!r}; model={actual!r}"
        )
    return expected


def _scalar(value: object) -> object:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(()).item()
    return value


def _metadata_bool_is_true(value: object) -> bool:
    scalar = _scalar(value)
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, str):
        return scalar.strip().lower() == "true"
    if isinstance(scalar, (int, np.integer)):
        return int(scalar) == 1
    return False


def cache_entry_error(cache_file: Path, *, backbone: str, expected_dim: int) -> str | None:
    try:
        with np.load(str(cache_file)) as data:
            files = set(data.files)
            if "emb" not in files:
                return "missing emb"

            emb = np.asarray(data["emb"], dtype=np.float32).reshape(-1)
            if emb.size != int(expected_dim):
                return f"dim={emb.size}, expected exactly {expected_dim}"

            for key in ("backbone", "embed_dim", "pretrained_required", "pretrained_loaded"):
                if key not in files:
                    return f"missing metadata field: {key}"

            cached_backbone = str(_scalar(data["backbone"]))
            if cached_backbone != str(backbone):
                return f"backbone={cached_backbone!r}, expected {str(backbone)!r}"

            cached_embed_dim = int(_scalar(data["embed_dim"]))
            if cached_embed_dim != int(expected_dim):
                return f"embed_dim={cached_embed_dim}, expected {int(expected_dim)}"

            if not _metadata_bool_is_true(data["pretrained_required"]):
                return "pretrained_required is not true"
            if not _metadata_bool_is_true(data["pretrained_loaded"]):
                return "pretrained_loaded is not true"

            if "expected_embed_dim" in files and int(_scalar(data["expected_embed_dim"])) != int(expected_dim):
                return f"expected_embed_dim={int(_scalar(data['expected_embed_dim']))}, expected {int(expected_dim)}"

            return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def cache_entry_is_valid(cache_file: Path, *, backbone: str, expected_dim: int) -> bool:
    return cache_entry_error(cache_file, backbone=backbone, expected_dim=expected_dim) is None
