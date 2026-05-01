from __future__ import annotations

import argparse
import json
import os
import hashlib
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

from src.fpbench.matchers.baseline_dl import BaselineDL, DLBaselineConfig
from src.fpbench.preprocess.preprocess import PreprocessConfig


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


def cache_file_for(cache_root: Path, path: str, cfg_json: str, strip_prefix: str) -> Path:
    key_src = canonical_path(path, strip_prefix) + "|" + cfg_json
    h = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    return cache_root / h[:2] / f"{h}.npz"


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
    ap.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50", "vit_base"])
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
    paths = [p for p, _ in path_and_capture]
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    cache_root = parse_file_uri(args.emb_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    dl_cfg = DLBaselineConfig(backbone=args.backbone, use_mask=(not args.no_mask))
    prep_cfg = PreprocessConfig(target_size=512)
    device = args.device.strip() or None
    model = BaselineDL(dl_cfg=dl_cfg, prep_cfg=prep_cfg, device=device)

    cfg_for_key = model.config_dict()
    cfg_for_key.pop("device", None)
    cfg_json = json.dumps(cfg_for_key, sort_keys=True, ensure_ascii=False)

    wrote = 0
    skipped = 0

    print("resolved_data_dir:", str(data_dir))
    print("manifest:", str(manifest))
    print("cache_dir:", str(cache_root))
    print("n_paths:", len(paths))

    for i, (p, capture) in enumerate(path_and_capture, 1):
        cf = cache_file_for(cache_root, p, cfg_json, args.cache_strip_prefix)

        if cf.exists():
            skipped += 1
        else:
            src_path = resolve_input_path(p)
            if not src_path.exists():
                raise FileNotFoundError(f"Missing source image from manifest: {src_path}")

            emb, _ = model.embed_path(str(src_path), capture=capture)
            cf.parent.mkdir(parents=True, exist_ok=True)
            tmp = cf.with_suffix(".tmp.npz")
            np.savez_compressed(str(tmp), emb=emb.astype(np.float32))
            os.replace(str(tmp), str(cf))
            wrote += 1

        if i % 200 == 0 or i == len(paths):
            print(f"[{args.dataset}] {i}/{len(paths)} | wrote={wrote} | skipped={skipped}")

    print("DONE.")
    print("resolved_data_dir:", str(data_dir))
    print("manifest:", str(manifest))
    print("cache_dir:", str(cache_root))
    print("wrote:", wrote, "skipped:", skipped)
    print("cfg:", cfg_for_key)


if __name__ == "__main__":
    main()