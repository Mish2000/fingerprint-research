from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.ingest.optional_raw_only_utils import (
    DEFAULT_NEG_PER_POS,
    DEFAULT_SEED,
    assign_split,
    build_basic_sanity,
    choose_one_per_key,
    repo_root_from_here,
    split_by_subject,
    stats_with_pairs,
    write_nested_pairs_bundle,
    write_pair_metadata,
    write_pairs_csvs,
    write_split_json,
    write_stats_and_sanity,
)

DATASET = "l3_sf_v2"
DEFAULT_FINGER_COL = "frgp"
DEFAULT_POSITIVE_POLICY = "same_subject_same_finger_within_subset_synthetic_level3"
DEFAULT_NEGATIVE_POLICY = "same_finger_other_subject_within_subset_synthetic_level3_same_split"
IMAGE_COLUMNS = [
    "dataset",
    "capture",
    "subject_id",
    "impression",
    "ppi",
    "frgp",
    "path",
    "split",
    "sample_id",
    "session",
    "source_modality",
    "subset",
    "subset_index",
    "subject_local_id",
]
SUBSET_RE = re.compile(r"^R(?P<subset>\d+)$", re.IGNORECASE)
IMAGE_RE = re.compile(r"^(?P<subject>\d+)_(?P<finger>\d+)_(?P<sample>\d+)$")


def _resolve_dir(path: Optional[str], fallback_candidates: Iterable[Path]) -> Optional[Path]:
    if path:
        p = Path(path).expanduser().resolve()
        return p if p.exists() else p
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_l3_dirs(raw_root: Path, image_dir: Optional[str], pore_gt_dir: Optional[str]) -> tuple[Path, Optional[Path]]:
    candidate_roots = [raw_root]
    if raw_root.name != "L3_SF_V2":
        candidate_roots.extend([
            raw_root / "L3_SF_V2",
            raw_root / "l3_sf_v2",
        ])

    image_candidates = []
    pore_candidates = []
    for base in candidate_roots:
        image_candidates.extend([base / "L3SF_V2" / "L3-SF", base / "L3-SF", base])
        pore_candidates.extend([
            base / "L3SF_V2" / "Pore ground truth" / "Fingerprint Images",
            base / "Pore ground truth" / "Fingerprint Images",
        ])

    image_root = _resolve_dir(image_dir, image_candidates)
    pore_root = _resolve_dir(pore_gt_dir, pore_candidates)
    if image_root is None:
        raise FileNotFoundError(f"Could not resolve L3-SF v2 image directory under {raw_root}")
    return image_root, pore_root


def _subset_info(path: Path) -> tuple[Optional[str], Optional[int]]:
    for part in path.parts:
        m = SUBSET_RE.match(part)
        if m:
            idx = int(m.group("subset"))
            return f"R{idx}", idx
    return None, None


def build_manifest(image_root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        subset, subset_index = _subset_info(path)
        if subset is None or subset_index is None:
            continue
        m = IMAGE_RE.match(path.stem)
        if m is None:
            continue
        subject_local_id = int(m.group("subject"))
        finger_id = int(m.group("finger"))
        sample_id = int(m.group("sample"))
        subject_id = subset_index * 100000 + subject_local_id
        rows.append(
            {
                "dataset": DATASET,
                "capture": "contact_based",
                "subject_id": int(subject_id),
                "impression": f"capture_{sample_id:02d}",
                "ppi": 0,
                "frgp": int(finger_id),
                "path": str(path.resolve()),
                "split": None,
                "sample_id": int(sample_id),
                "session": int(subset_index),
                "source_modality": "synthetic_level3",
                "subset": subset,
                "subset_index": int(subset_index),
                "subject_local_id": int(subject_local_id),
            }
        )
    if not rows:
        return pd.DataFrame(columns=IMAGE_COLUMNS)
    df = pd.DataFrame(rows)
    return df.sort_values(["subset_index", "subject_local_id", "frgp", "sample_id", "path"]).reset_index(drop=True)


def count_pore_ground_truth(pore_root: Optional[Path]) -> int:
    if pore_root is None or not pore_root.exists():
        return 0
    return sum(1 for p in pore_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".bmp", ".jpeg", ".tif", ".tiff"})


def _canonical_pairs(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["pair_id", "label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"])
    return pd.DataFrame(rows, columns=["label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"])


def make_verification_pairs(df: pd.DataFrame, *, max_pos_per_group: int, seed: int, neg_per_pos: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = choose_one_per_key(
        df[df["source_modality"] == "synthetic_level3"],
        ["subject_id", "frgp", "impression", "subset", "split"],
        sort_cols=["subset_index", "subject_local_id", "frgp", "sample_id", "path"],
    )

    pos_rows: List[dict] = []
    for (split_name, subset, subject_id, frgp), group in base.groupby(["split", "subset", "subject_id", "frgp"]):
        paths = group.sort_values(["sample_id", "path"])["path"].tolist()
        combos: List[tuple[str, str]] = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                combos.append((paths[i], paths[j]))
        for path_a, path_b in combos[:max_pos_per_group]:
            pos_rows.append(
                {
                    "label": 1,
                    "split": str(split_name),
                    "subject_a": int(subject_id),
                    "subject_b": int(subject_id),
                    "frgp": int(frgp),
                    "path_a": path_a,
                    "path_b": path_b,
                }
            )
    pos = _canonical_pairs(pos_rows)

    rng = random.Random(seed)
    neg_rows: List[dict] = []
    pool = {}
    path_to_subset = {}
    path_to_frgp = {}
    for (split_name, subset, frgp), group in base.groupby(["split", "subset", "frgp"]):
        pool[(str(split_name), str(subset), int(frgp))] = [(int(row.subject_id), str(row.path)) for row in group[["subject_id", "path"]].itertuples(index=False)]
        for row in group[["path", "subset", "frgp"]].itertuples(index=False):
            path_to_subset[str(row.path)] = str(row.subset)
            path_to_frgp[str(row.path)] = int(row.frgp)

    for row in pos.itertuples(index=False):
        subset = path_to_subset.get(str(row.path_a), "R0")
        frgp = path_to_frgp.get(str(row.path_a), int(row.frgp))
        candidates = [item for item in pool.get((str(row.split), str(subset), int(frgp)), []) if int(item[0]) != int(row.subject_a)]
        if not candidates:
            continue
        for _ in range(int(neg_per_pos)):
            subject_b, path_b = rng.choice(candidates)
            neg_rows.append(
                {
                    "label": 0,
                    "split": str(row.split),
                    "subject_a": int(row.subject_a),
                    "subject_b": int(subject_b),
                    "frgp": int(frgp),
                    "path_a": str(row.path_a),
                    "path_b": str(path_b),
                }
            )
    return pos, _canonical_pairs(neg_rows)


def main() -> None:
    rr = repo_root_from_here()
    ap = argparse.ArgumentParser(description="Prepare optional L3-SF v2 manifest + pair bundle under data/manifests.")
    ap.add_argument("--raw_root", type=str, default=str(rr / "data" / "raw" / "L3_SF_V2"))
    ap.add_argument("--image_dir", type=str, default=None)
    ap.add_argument("--pore_gt_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=str(rr / "data" / "manifests" / DATASET))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--neg_per_pos", type=int, default=DEFAULT_NEG_PER_POS)
    ap.add_argument("--max_pos_per_group", type=int, default=10)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    image_root, pore_root = resolve_l3_dirs(raw_root, args.image_dir, args.pore_gt_dir)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_manifest(image_root)
    if df.empty:
        raise RuntimeError("No L3-SF v2 rows parsed. Check input directories.")
    pore_gt_files = count_pore_ground_truth(pore_root)

    split_map = split_by_subject(df, args.seed, args.train_ratio, args.val_ratio)
    write_split_json(out_dir, split_map)
    df = assign_split(df, split_map)
    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    pos, neg = make_verification_pairs(df, max_pos_per_group=int(args.max_pos_per_group), seed=int(args.seed), neg_per_pos=int(args.neg_per_pos))
    write_pairs_csvs(out_dir, dataset=DATASET, pos=pos, neg=neg)
    write_nested_pairs_bundle(
        out_dir,
        split_map,
        seed=int(args.seed),
        neg_per_pos=int(args.neg_per_pos),
        manifest_path=manifest_path,
        positive_pair_policy=DEFAULT_POSITIVE_POLICY,
        negative_pair_policy=DEFAULT_NEGATIVE_POLICY,
        finger_col=DEFAULT_FINGER_COL,
        resolved_data_dir=image_root,
        same_finger_policy=True,
        max_pos_per_finger=int(args.max_pos_per_group),
        pair_mode="within_subset_verification",
    )
    write_pair_metadata(
        out_dir,
        dataset=DATASET,
        seed=int(args.seed),
        neg_per_pos=int(args.neg_per_pos),
        finger_col=DEFAULT_FINGER_COL,
        positive_pair_policy=DEFAULT_POSITIVE_POLICY,
        negative_pair_policy=DEFAULT_NEGATIVE_POLICY,
        extra={
            "raw_root": str(raw_root),
            "image_root": str(image_root),
            "pore_gt_dir": str(pore_root) if pore_root is not None else None,
            "protocol": "verification",
            "pair_mode": "within_subset_verification",
            "max_pos_per_group": int(args.max_pos_per_group),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "subset_policy": "subject_id is namespaced by subset so repeated local IDs across R1-R5 cannot leak across splits or create invalid cross-domain positives",
            "notes": "L3-SF v2 is emitted as a synthetic single-modality manifest. Pore ground-truth images remain sidecars for auxiliary supervision and sanity only.",
        },
    )
    stats = stats_with_pairs(
        df,
        pos,
        neg,
        extra={
            "protocol": "verification",
            "pair_eligible_source_modalities": ["synthetic_level3"],
            "subset_count": int(df["subset"].nunique()),
            "pore_ground_truth_sidecars": int(pore_gt_files),
            "unique_local_subjects": int(df["subject_local_id"].nunique()),
        },
    )
    sanity = build_basic_sanity(df, split_map, pos, neg)
    write_stats_and_sanity(
        out_dir,
        dataset=DATASET,
        raw_root=raw_root,
        manifest_df=df,
        pos_df=pd.read_csv(out_dir / "pairs_pos.csv"),
        neg_df=pd.read_csv(out_dir / "pairs_neg.csv"),
        stats=stats,
        extra_manifest={"protocol": "verification", "pair_eligible_source_modalities": ["synthetic_level3"]},
        extra_checks={"basic_sanity": sanity},
    )

    print(json.dumps({"dataset": DATASET, "rows": len(df), "stats": stats, "sanity": sanity}, indent=2))


if __name__ == "__main__":
    main()
