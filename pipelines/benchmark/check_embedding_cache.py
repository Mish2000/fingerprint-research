from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd


def project_root() -> Path:
    env = os.environ.get("FPRJ_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
sys.path.insert(0, str(ROOT))

from src.fpbench.matchers.baseline_dl import DLBaselineConfig
from src.fpbench.preprocess.preprocess import PreprocessConfig


def resolve_path(s: str) -> Path:
    if s.startswith("file:"):
        s = s[len("file:"):]
        if len(s) >= 3 and s[0] == "/" and s[2] == ":":
            s = s[1:]
    p = Path(s).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def resolve_dataset_dir(input_dir: Optional[Path], dataset: str) -> Path:
    candidates = []

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

    uniq_candidates = []
    seen = set()
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Check persistent embedding cache coverage for a dataset.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--emb_cache_dir", required=True)
    ap.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50", "vit_base"])
    ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--cache_strip_prefix", default="")
    args = ap.parse_args()

    input_dir = resolve_path(args.data_dir) if args.data_dir else None
    data_dir = resolve_dataset_dir(input_dir, args.dataset)
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    df = pd.read_csv(manifest)
    path_col = "path" if "path" in df.columns else "file_path" if "file_path" in df.columns else None
    if path_col is None:
        raise ValueError("manifest.csv must contain either 'path' or 'file_path' column.")

    paths = [str(p) for p in df[path_col].astype(str).tolist() if str(p).strip()]
    paths = list(dict.fromkeys(paths))

    cache_root = resolve_path(args.emb_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    dl_cfg = DLBaselineConfig(backbone=args.backbone, use_mask=(not args.no_mask))
    prep_cfg = PreprocessConfig(target_size=512)
    embed_dim = {"resnet18": 512, "resnet50": 2048, "vit_base": 768}[args.backbone]

    cfg_for_key = {
        "dl_cfg": asdict(dl_cfg),
        "prep_cfg": asdict(prep_cfg),
        "embed_dim": embed_dim,
    }
    cfg_json = json.dumps(cfg_for_key, sort_keys=True, ensure_ascii=False)

    ready = 0
    missing = 0
    sample_missing_cache_files = []
    sample_missing_source_paths = []

    for p in paths:
        cf = cache_file_for(cache_root, p, cfg_json, args.cache_strip_prefix)
        if cf.exists():
            ready += 1
        else:
            missing += 1
            if len(sample_missing_cache_files) < 10:
                sample_missing_cache_files.append(str(cf))
            if len(sample_missing_source_paths) < 10:
                sample_missing_source_paths.append(str(p))

    tmp_files = [str(p) for p in cache_root.rglob("*.tmp.npz")]

    coverage_pct = 100.0 * ready / max(1, len(paths))
    summary = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "manifest": str(manifest),
        "cache_dir": str(cache_root),
        "backbone": args.backbone,
        "use_mask": bool(not args.no_mask),
        "total_paths": len(paths),
        "ready": ready,
        "missing": missing,
        "coverage_pct": coverage_pct,
        "tmp_files_count": len(tmp_files),
        "tmp_files_sample": tmp_files[:10],
        "sample_missing_cache_files": sample_missing_cache_files,
        "sample_missing_source_paths": sample_missing_source_paths,
        "cfg": cfg_for_key,
    }

    suffix = "nomask" if args.no_mask else "mask"
    out_json = cache_root / f"cache_status_{args.dataset}_{args.backbone}_{suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())