from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def project_root() -> Path:
    env = os.environ.get("FPRJ_ROOT", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


ROOT = project_root()

import sys
sys.path.insert(0, str(ROOT))

from src.fpbench.matchers.baseline_dl import BaselineDL, DLBaselineConfig, expected_embed_dim_for_backbone
from src.fpbench.preprocess.preprocess import PreprocessConfig
from pipelines.benchmark.embedding_cache import (
    assert_cache_key_config_matches_model,
    cache_entry_is_valid,
    cache_file_for,
)


def normalize_capture(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    aliases = {
        "plain": "plain",
        "roll": "roll",
        "rolled": "roll",
        "contactless": "contactless",
        "contact-less": "contactless",
        "contact_less": "contactless",
        "contact_based": "contact_based",
        "contact-based": "contact_based",
        "contactbased": "contact_based",
    }
    if s not in aliases:
        raise ValueError(f"Unsupported capture label in manifest: {raw}")
    return aliases[s]


def parse_file_uri(p: str) -> Path:
    if p.startswith("file:"):
        p = p[len("file:"):]
        if p.startswith("/"):
            p = p[1:]
    return Path(p.replace("/", "\\")).expanduser().resolve()


def resolve_dataset_dir(input_dir: Optional[Path], dataset: str) -> Path:
    """
    Supports both:
      - data/processed/<dataset>
      - data/manifests/<dataset>

    Returns the first directory that actually contains manifest.csv.
    """
    candidates: list[Path] = []

    if input_dir is not None:
        candidates.append(input_dir)

        try:
            parent_name = input_dir.parent.name.lower()
            if parent_name == "processed":
                candidates.append(input_dir.parent.parent / "manifests" / input_dir.name)
            elif parent_name == "manifests":
                candidates.append(input_dir.parent.parent / "processed" / input_dir.name)
        except Exception:
            pass

    candidates.append(ROOT / "data" / "processed" / dataset)
    candidates.append(ROOT / "data" / "manifests" / dataset)

    uniq_candidates: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            uniq_candidates.append(c)

    for c in uniq_candidates:
        if (c / "manifest.csv").exists():
            return c

    checked = [str(c) for c in uniq_candidates]
    raise FileNotFoundError(
        "Could not locate dataset directory containing manifest.csv. "
        f"Checked: {checked}"
    )


def resolve_input_path(path_str: str) -> Path:
    s = str(path_str).strip()
    if s.startswith("file:"):
        return parse_file_uri(s)

    p = Path(s).expanduser()
    if p.is_absolute():
        return p.resolve()

    return (ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser("Warm persistent embedding cache for BaselineDL.")
    ap.add_argument("--dataset", required=True, help="e.g. nist_sd300b / nist_sd300c")
    ap.add_argument(
        "--data_dir",
        default="",
        help="Optional dataset dir. Supports either data/processed/<dataset> or data/manifests/<dataset>.",
    )
    ap.add_argument("--emb_cache_dir", required=True)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50", "vit_base"])
    ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--device", default="", help="cuda|cpu or empty=auto")
    ap.add_argument("--cache_strip_prefix", default="")
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    args = ap.parse_args()

    input_dir = Path(args.data_dir).expanduser() if args.data_dir else (ROOT / "data" / "processed" / args.dataset)
    data_dir = resolve_dataset_dir(input_dir, args.dataset)
    manifest = data_dir / "manifest.csv"

    df = pd.read_csv(manifest)
    col = "path" if "path" in df.columns else "file_path" if "file_path" in df.columns else None
    if col is None:
        raise ValueError("manifest.csv must contain either 'path' or 'file_path' column.")

    capture_col = "capture" if "capture" in df.columns else None
    path_and_capture = []
    for _, row in df.iterrows():
        p = str(row[col]).strip()
        if not p:
            continue
        cap = normalize_capture(row[capture_col]) if capture_col else None
        path_and_capture.append((p, cap))
    path_and_capture = list(dict.fromkeys(path_and_capture))  # stable unique
    if args.limit and args.limit > 0:
        path_and_capture = path_and_capture[: args.limit]
    paths = [p for p, _ in path_and_capture]

    cache_root = parse_file_uri(args.emb_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    expected_dim = expected_embed_dim_for_backbone(args.backbone)

    dl_cfg = DLBaselineConfig(backbone=args.backbone, use_mask=(not args.no_mask))
    prep_cfg = PreprocessConfig(target_size=512)
    device = args.device.strip() or None
    model = BaselineDL(dl_cfg=dl_cfg, prep_cfg=prep_cfg, device=device)
    actual_dim = int(model.config_dict().get("embed_dim", -1))
    if actual_dim != int(expected_dim):
        raise RuntimeError(f"Loaded backbone={args.backbone!r} produced embed_dim={actual_dim}, expected {expected_dim}")

    cfg_for_key = assert_cache_key_config_matches_model(model=model, dl_cfg=dl_cfg, prep_cfg=prep_cfg)
    cfg_json = json.dumps(cfg_for_key, sort_keys=True, ensure_ascii=False)

    wrote = 0
    skipped = 0
    invalid = 0

    print("resolved_data_dir:", str(data_dir))
    print("manifest:", str(manifest))
    print("cache_dir:", str(cache_root))
    print("n_paths:", len(paths))

    for i, (p, capture) in enumerate(path_and_capture, 1):
        cf = cache_file_for(cache_root, p, cfg_json, args.cache_strip_prefix)

        if cf.exists() and cache_entry_is_valid(cf, backbone=args.backbone, expected_dim=expected_dim):
            skipped += 1
        else:
            if cf.exists():
                invalid += 1
            src_path = resolve_input_path(p)
            if not src_path.exists():
                raise FileNotFoundError(f"Missing source image from manifest: {src_path}")

            emb, _ = model.embed_path(str(src_path), capture=capture)
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb.size != int(expected_dim):
                raise RuntimeError(
                    f"Embedding for backbone={args.backbone!r} has dim={emb.size}, expected exactly {expected_dim}"
                )
            cf.parent.mkdir(parents=True, exist_ok=True)
            tmp = cf.with_suffix(".tmp.npz")
            np.savez_compressed(
                str(tmp),
                emb=emb.astype(np.float32),
                backbone=np.array(args.backbone),
                embed_dim=np.array(int(expected_dim)),
                expected_embed_dim=np.array(int(expected_dim)),
                pretrained_required=np.array(True),
                pretrained_loaded=np.array(True),
            )
            os.replace(str(tmp), str(cf))
            wrote += 1

        if i % 200 == 0 or i == len(paths):
            print(f"[{args.dataset}] {i}/{len(paths)} | wrote={wrote} | skipped={skipped} | invalid={invalid}")

    print("DONE.")
    print("resolved_data_dir:", str(data_dir))
    print("manifest:", str(manifest))
    print("cache_dir:", str(cache_root))
    print("wrote:", wrote, "skipped:", skipped, "invalid:", invalid)
    print("cfg:", cfg_for_key)


if __name__ == "__main__":
    main()
