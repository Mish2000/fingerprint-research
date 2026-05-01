from __future__ import annotations

import sys
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.ingest.pair_bundle_utils import (
    canonicalize_pairs_df,
    build_pairs_split_build_meta,
    build_split_subjects_metadata,
    validate_canonical_pairs_df,
    validate_pairs_split_build_meta,
    validate_split_subjects_metadata,
    write_json,
)

DATASETS = {
    "nist_sd300b": {
        "seed": 42,
        "neg_per_pos": 3,
        "impostors_per_pos": 3,
        "finger_col": "frgp",
        "positive_pair_policy": "same_subject_same_finger_plain_to_roll",
        "negative_pair_policy": "same_finger_other_subject_same_split",
        "same_finger_policy": True,
        "max_pos_per_subject": 5000,
        "max_pos_per_finger": 500,
        "pair_mode": "plain_to_roll",
    },
    "nist_sd300c": {
        "seed": 42,
        "neg_per_pos": 3,
        "impostors_per_pos": 3,
        "finger_col": "frgp",
        "positive_pair_policy": "same_subject_same_finger_plain_to_roll",
        "negative_pair_policy": "same_finger_other_subject_same_split",
        "same_finger_policy": True,
        "max_pos_per_subject": 5000,
        "max_pos_per_finger": 500,
        "pair_mode": "plain_to_roll",
    },
    "polyu_cross": {
        "seed": 42,
        "neg_per_pos": 3,
        "impostors_per_pos": 3,
        "finger_col": "frgp",
        "positive_pair_policy": "same_subject_same_finger_contactless_to_contact_based",
        "negative_pair_policy": "same_finger_other_subject_same_split",
        "same_finger_policy": True,
        "max_pos_per_subject": 12,
        "pair_mode": "cross_capture_probe_to_gallery",
    },
}


def _copy_to_nested(df: pd.DataFrame, base: Path, split: str) -> None:
    nested = base / "pairs" / f"pairs_{split}.csv"
    nested.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(nested, index=False)


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {[str(p) for p in paths]}")


def regenerate_dataset(ds: str, cfg: dict) -> None:
    base = ROOT / "data" / "manifests" / ds
    manifest_path = base / "manifest.csv"
    split_path = base / "split.json"
    if not manifest_path.exists() or not split_path.exists():
        raise FileNotFoundError(f"Missing manifest or split for {ds}: {base}")

    splits = json.loads(split_path.read_text(encoding="utf-8"))

    for split in ("train", "val", "test"):
        src = _first_existing(base / f"pairs_{split}.csv", base / "pairs" / f"pairs_{split}.csv")
        df = pd.read_csv(src)
        canon = canonicalize_pairs_df(df, split=split)
        canon = validate_canonical_pairs_df(canon, context=f"{ds}/{split} canonical pairs", expected_split=split, require_exact_columns=True)
        canon.to_csv(base / f"pairs_{split}.csv", index=False)
        _copy_to_nested(canon, base, split)

    for name in ("pairs_pos.csv", "pairs_neg.csv"):
        p = base / name
        if p.exists():
            df = pd.read_csv(p)
            split = None
            if "split" in df.columns and not df["split"].isna().all():
                split = None
            elif len(df):
                split = "mixed"
            canon = canonicalize_pairs_df(df, split=split) if len(df) else canonicalize_pairs_df(pd.DataFrame(columns=["label","subject_a","subject_b","frgp","path_a","path_b"]), split="mixed")
            canon.to_csv(p, index=False)

    split_meta = build_split_subjects_metadata(
        splits=splits,
        seed=cfg["seed"],
        neg_per_pos=cfg["neg_per_pos"],
        impostors_per_pos=cfg["impostors_per_pos"],
        same_finger_policy=cfg["same_finger_policy"],
        negative_pair_policy=cfg["negative_pair_policy"],
        positive_pair_policy=cfg["positive_pair_policy"],
        finger_col=cfg["finger_col"],
        resolved_data_dir=base,
        manifest_path=manifest_path,
        max_pos_per_subject=cfg.get("max_pos_per_subject"),
        max_pos_per_finger=cfg.get("max_pos_per_finger"),
        pair_mode=cfg.get("pair_mode"),
    )
    validate_split_subjects_metadata(split_meta, context=f"{ds} split_subjects metadata")
    write_json(base / "pairs" / "split_subjects.json", split_meta)

    meta_existing = {}
    meta_path = base / "pairs_split_build.meta.json"
    if meta_path.exists():
        try:
            meta_existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta_existing = {}
    meta = build_pairs_split_build_meta(
        dataset=ds,
        seed=cfg["seed"],
        neg_per_pos=cfg["neg_per_pos"],
        impostors_per_pos=cfg["impostors_per_pos"],
        finger_col=cfg["finger_col"],
        positive_pair_policy=cfg["positive_pair_policy"],
        negative_pair_policy=cfg["negative_pair_policy"],
        extra=meta_existing,
    )
    validate_pairs_split_build_meta(meta, context=f"{ds} pairs_split_build metadata")
    write_json(meta_path, meta)
    print(f"[OK] regenerated {ds}")


def main() -> None:
    for ds, cfg in DATASETS.items():
        regenerate_dataset(ds, cfg)


if __name__ == "__main__":
    main()
