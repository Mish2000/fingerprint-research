from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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
    infer_session,
    iter_files,
    iter_images,
    repo_root_from_here,
    split_by_subject,
    stats_with_pairs,
    write_nested_pairs_bundle,
    write_pair_metadata,
    write_pairs_csvs,
    write_split_json,
    write_stats_and_sanity,
)

DATASET = "unsw_2d3d"
DEFAULT_FINGER_COL = "frgp"
PROTOCOLS = {
    "cross_modality": {
        "positive": "same_subject_same_finger_optical_2d_to_reconstructed_3d",
        "negative": "same_finger_other_subject_optical_2d_to_reconstructed_3d_same_split",
        "pair_mode": "cross_modality_probe_to_gallery",
        "same_finger_policy": True,
        "left_modalities": {"optical_2d"},
        "right_modalities": {"reconstructed_3d"},
    },
    "verification_2d": {
        "positive": "same_subject_same_finger_within_optical_2d",
        "negative": "same_finger_other_subject_within_optical_2d_same_split",
        "pair_mode": "within_modality_verification",
        "same_finger_policy": True,
        "modalities": {"optical_2d"},
    },
    "verification_3d": {
        "positive": "same_subject_same_finger_within_reconstructed_3d",
        "negative": "same_finger_other_subject_within_reconstructed_3d_same_split",
        "pair_mode": "within_modality_verification",
        "same_finger_policy": True,
        "modalities": {"reconstructed_3d"},
    },
}
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
    "variant",
    "variant_rank",
    "frame_id",
]
TWO_D_RE = re.compile(r"^(?P<subject>\d+)_(?P<finger>\d+)_(?P<sample>\d+)$", re.IGNORECASE)
THREE_D_RE = re.compile(r"^SIRE-(?P<subject>\d+)_(?P<finger>\d+)_(?P<sample>\d+)(?:_(?P<variant>[A-Za-z0-9]+))?$", re.IGNORECASE)
RAW_RE = re.compile(r"^(?P<subject>\d+)_(?P<finger>\d+)_(?P<sample>\d+)_(?P<frame>\d+)$", re.IGNORECASE)

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p



def _resolve_dir(path: Optional[str], fallback_candidates: Iterable[Path]) -> Optional[Path]:
    if path:
        p = Path(path).expanduser().resolve()
        return p if p.exists() else p
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_unsw_dirs(raw_root: Path, two_d_dir: Optional[str], three_d_dirs: Sequence[str]) -> tuple[Path, List[Path]]:
    candidate_roots = [raw_root]
    if raw_root.name.lower() != "unsw 3d":
        candidate_roots.extend([
            raw_root / "UNSW 3D",
            raw_root / "unsw_2d3d",
            raw_root / "UNSW_2D3D",
        ])

    two_d_candidates = []
    for base in candidate_roots:
        two_d_candidates.extend([
            base / "2D_Database" / "2D Database",
            base / "2D_Database",
            base / "2D Database",
        ])

    two_d = _resolve_dir(
        two_d_dir,
        two_d_candidates,
    )
    if two_d is None:
        raise FileNotFoundError(f"Could not resolve UNSW 2D directory under {raw_root}")

    resolved_three_d: List[Path] = []
    scan_roots = candidate_roots
    if three_d_dirs:
        for entry in three_d_dirs:
            resolved_three_d.append(Path(entry).expanduser().resolve())
    else:
        for base in scan_roots:
            for child in sorted(base.iterdir()) if base.exists() else []:
                if child.is_dir() and re.fullmatch(r"DS\d+", child.name, flags=re.IGNORECASE):
                    resolved_three_d.append(child.resolve())
    if not resolved_three_d:
        fallback_candidates = []
        for base in scan_roots:
            fallback_candidates.extend([base / "3D_Database", base / "3D Database"])
        fallback = _resolve_dir(None, fallback_candidates)
        if fallback is not None:
            resolved_three_d.append(fallback)
    if not resolved_three_d:
        raise FileNotFoundError(f"Could not resolve UNSW 3D DS directories under {raw_root}")
    return two_d, resolved_three_d


def _make_row(subject_id: int, finger_id: int, sample_id: int, *, capture: str, path: Path, source_modality: str, variant: str, variant_rank: int, frame_id: int = 0) -> dict:
    return {
        "dataset": DATASET,
        "capture": capture,
        "subject_id": int(subject_id),
        "impression": f"capture_{int(sample_id):02d}",
        "ppi": 0,
        "frgp": int(finger_id),
        "path": str(path.resolve()),
        "split": None,
        "sample_id": int(sample_id),
        "session": infer_session(path),
        "source_modality": source_modality,
        "variant": variant,
        "variant_rank": int(variant_rank),
        "frame_id": int(frame_id),
    }


def build_manifest(two_d_dir: Path, three_d_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: List[dict] = []

    for path in iter_images(two_d_dir):
        m = TWO_D_RE.match(path.stem)
        if m is None:
            continue
        rows.append(
            _make_row(
                int(m.group("subject")),
                int(m.group("finger")),
                int(m.group("sample")),
                capture="contact_based",
                path=path,
                source_modality="optical_2d",
                variant="primary",
                variant_rank=0,
            )
        )

    for ds_dir in three_d_dirs:
        for path in iter_images(ds_dir):
            is_raw_frame = path.parent.name.lower() == "raw"
            if is_raw_frame:
                m = RAW_RE.match(path.stem)
                if m is None:
                    continue
                rows.append(
                    _make_row(
                        int(m.group("subject")),
                        int(m.group("finger")),
                        int(m.group("sample")),
                        capture="contactless",
                        path=path,
                        source_modality="reconstruction_intermediate",
                        variant="raw",
                        variant_rank=3,
                        frame_id=int(m.group("frame")),
                    )
                )
                continue

            m = THREE_D_RE.match(path.stem)
            if m is None:
                continue
            variant = str(m.group("variant") or "primary").lower()
            source_modality = "reconstructed_3d" if variant == "primary" else "derived_3d_variant"
            variant_rank = 0 if variant == "primary" else 2
            rows.append(
                _make_row(
                    int(m.group("subject")),
                    int(m.group("finger")),
                    int(m.group("sample")),
                    capture="contactless",
                    path=path,
                    source_modality=source_modality,
                    variant=variant,
                    variant_rank=variant_rank,
                )
            )

    if not rows:
        return pd.DataFrame(columns=IMAGE_COLUMNS)

    df = pd.DataFrame(rows)
    return df.sort_values(["subject_id", "frgp", "source_modality", "sample_id", "variant_rank", "frame_id", "path"]).reset_index(drop=True)


def _canonical_pairs(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["pair_id", "label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"])
    return pd.DataFrame(rows, columns=["label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"])

def filter_protocol_eligible_manifest(df: pd.DataFrame, protocol_name: str) -> pd.DataFrame:
    protocol = PROTOCOLS[protocol_name]

    if protocol_name == 'cross_modality':
        left = df[df['source_modality'].isin(protocol['left_modalities'])][['subject_id', 'frgp']].drop_duplicates()
        right = df[df['source_modality'].isin(protocol['right_modalities'])][['subject_id', 'frgp']].drop_duplicates()
        eligible = left.merge(right, on=['subject_id', 'frgp'], how='inner')
        if eligible.empty:
            left_count = int(len(left))
            right_count = int(len(right))
            raise RuntimeError(
                'No UNSW subject/finger combinations are shared between optical_2d and reconstructed_3d. '
                f'Parsed {left_count} optical_2d subject/finger keys and {right_count} reconstructed_3d keys. '
                'This usually means BMP files were skipped by the image iterator or the selected DS folders do not match the 2D database.'
            )
        filtered = df.merge(eligible.assign(_eligible=1), on=['subject_id', 'frgp'], how='inner').drop(columns=['_eligible'])
        return filtered.reset_index(drop=True)

    return df


def make_cross_modality_pairs(
    df: pd.DataFrame,
    *,
    left_modalities: set[str],
    right_modalities: set[str],
    max_pos_per_group: int,
    seed: int,
    neg_per_pos: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = choose_one_per_key(
        df[df["source_modality"].isin(left_modalities)],
        ["subject_id", "frgp", "impression", "source_modality", "split"],
        sort_cols=["subject_id", "frgp", "sample_id", "path"],
    )
    right = choose_one_per_key(
        df[df["source_modality"].isin(right_modalities)],
        ["subject_id", "frgp", "impression", "source_modality", "split"],
        sort_cols=["subject_id", "frgp", "sample_id", "path"],
    )

    pos_rows: List[dict] = []
    for (split_name, subject_id, frgp), group_left in left.groupby(["split", "subject_id", "frgp"]):
        group_right = right[(right["split"] == split_name) & (right["subject_id"] == subject_id) & (right["frgp"] == frgp)]
        if group_right.empty:
            continue
        combos: List[tuple[str, str]] = []
        for path_a in group_left.sort_values(["sample_id", "path"])["path"].tolist():
            for path_b in group_right.sort_values(["sample_id", "path"])["path"].tolist():
                if path_a != path_b:
                    combos.append((path_a, path_b))
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
    right_pool: Dict[tuple[str, int], List[tuple[int, str]]] = {}
    for (split_name, frgp), group in right.groupby(["split", "frgp"]):
        right_pool[(str(split_name), int(frgp))] = [(int(row.subject_id), str(row.path)) for row in group[["subject_id", "path"]].itertuples(index=False)]

    neg_rows: List[dict] = []
    for row in pos.itertuples(index=False):
        candidates = [item for item in right_pool.get((str(row.split), int(row.frgp)), []) if int(item[0]) != int(row.subject_a)]
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
                    "frgp": int(row.frgp),
                    "path_a": str(row.path_a),
                    "path_b": str(path_b),
                }
            )
    return pos, _canonical_pairs(neg_rows)


def make_within_modality_pairs(
    df: pd.DataFrame,
    *,
    modalities: set[str],
    max_pos_per_group: int,
    seed: int,
    neg_per_pos: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = choose_one_per_key(
        df[df["source_modality"].isin(modalities)],
        ["subject_id", "frgp", "impression", "source_modality", "split"],
        sort_cols=["subject_id", "frgp", "sample_id", "path"],
    )

    pos_rows: List[dict] = []
    for (split_name, subject_id, frgp, source_modality), group in base.groupby(["split", "subject_id", "frgp", "source_modality"]):
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
    pool: Dict[tuple[str, int, str], List[tuple[int, str]]] = {}
    for (split_name, frgp, source_modality), group in base.groupby(["split", "frgp", "source_modality"]):
        pool[(str(split_name), int(frgp), str(source_modality))] = [(int(row.subject_id), str(row.path)) for row in group[["subject_id", "path"]].itertuples(index=False)]
    path_to_modality = {str(row.path): str(row.source_modality) for row in base[["path", "source_modality"]].itertuples(index=False)}

    neg_rows: List[dict] = []
    for row in pos.itertuples(index=False):
        modality = path_to_modality.get(str(row.path_a))
        candidates = [item for item in pool.get((str(row.split), int(row.frgp), str(modality)), []) if int(item[0]) != int(row.subject_a)]
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
                    "frgp": int(row.frgp),
                    "path_a": str(row.path_a),
                    "path_b": str(path_b),
                }
            )
    return pos, _canonical_pairs(neg_rows)


def main() -> None:
    rr = repo_root_from_here()
    ap = argparse.ArgumentParser(description="Prepare optional UNSW 2D/3D manifest + pair bundle under data/manifests.")
    ap.add_argument("--raw_root", type=str, default=str(rr / "data" / "raw" / "UNSW 3D"))
    ap.add_argument("--two_d_dir", type=str, default=None)
    ap.add_argument("--three_d_dir", dest="three_d_dirs", action="append", default=[])
    ap.add_argument("--out_dir", type=str, default=str(rr / "data" / "manifests" / DATASET))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--neg_per_pos", type=int, default=DEFAULT_NEG_PER_POS)
    ap.add_argument("--max_pos_per_group", type=int, default=8)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS.keys()), default="cross_modality")
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    two_d_dir, three_d_dirs = resolve_unsw_dirs(raw_root, args.two_d_dir, args.three_d_dirs)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_manifest(two_d_dir, three_d_dirs)
    if df.empty:
        raise RuntimeError("No UNSW rows parsed. Check input directories.")

    df = filter_protocol_eligible_manifest(df, args.protocol)
    if df.empty:
        raise RuntimeError(
            f"No protocol-eligible UNSW rows remained after filtering for {args.protocol}."
        )

    split_map = split_by_subject(df, args.seed, args.train_ratio, args.val_ratio)
    write_split_json(out_dir, split_map)
    df = assign_split(df, split_map)
    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    protocol = PROTOCOLS[args.protocol]
    if args.protocol == "cross_modality":
        pos, neg = make_cross_modality_pairs(
            df,
            left_modalities=set(protocol["left_modalities"]),
            right_modalities=set(protocol["right_modalities"]),
            max_pos_per_group=int(args.max_pos_per_group),
            seed=int(args.seed),
            neg_per_pos=int(args.neg_per_pos),
        )
    else:
        pos, neg = make_within_modality_pairs(
            df,
            modalities=set(protocol["modalities"]),
            max_pos_per_group=int(args.max_pos_per_group),
            seed=int(args.seed),
            neg_per_pos=int(args.neg_per_pos),
        )

    write_pairs_csvs(out_dir, dataset=DATASET, pos=pos, neg=neg)
    write_nested_pairs_bundle(
        out_dir,
        split_map,
        seed=int(args.seed),
        neg_per_pos=int(args.neg_per_pos),
        manifest_path=manifest_path,
        positive_pair_policy=str(protocol["positive"]),
        negative_pair_policy=str(protocol["negative"]),
        finger_col=DEFAULT_FINGER_COL,
        resolved_data_dir=raw_root,
        same_finger_policy=bool(protocol["same_finger_policy"]),
        max_pos_per_finger=int(args.max_pos_per_group),
        pair_mode=str(protocol["pair_mode"]),
    )
    write_pair_metadata(
        out_dir,
        dataset=DATASET,
        seed=int(args.seed),
        neg_per_pos=int(args.neg_per_pos),
        finger_col=DEFAULT_FINGER_COL,
        positive_pair_policy=str(protocol["positive"]),
        negative_pair_policy=str(protocol["negative"]),
        extra={
            "raw_root": str(raw_root),
            "two_d_dir": str(two_d_dir),
            "three_d_dirs": [str(p) for p in three_d_dirs],
            "protocol": str(args.protocol),
            "pair_mode": str(protocol["pair_mode"]),
            "max_pos_per_group": int(args.max_pos_per_group),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "notes": "UNSW optional ingest keeps optical 2D, primary reconstructed 3D, derived variants, and raw reconstruction intermediates in one manifest. Only protocol-eligible modalities are emitted into the canonical pair bundle.",
        },
    )
    stats = stats_with_pairs(
        df,
        pos,
        neg,
        extra={
            "protocol": str(args.protocol),
            "pair_eligible_source_modalities": sorted(list(protocol.get("modalities") or (set(protocol["left_modalities"]) | set(protocol["right_modalities"])))),
            "two_d_rows": int((df["source_modality"] == "optical_2d").sum()),
            "reconstructed_3d_rows": int((df["source_modality"] == "reconstructed_3d").sum()),
            "derived_3d_variant_rows": int((df["source_modality"] == "derived_3d_variant").sum()),
            "reconstruction_intermediate_rows": int((df["source_modality"] == "reconstruction_intermediate").sum()),
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
        extra_manifest={"protocol": str(args.protocol), "pair_eligible_source_modalities": sorted(list(protocol.get("modalities") or (set(protocol["left_modalities"]) | set(protocol["right_modalities"]))))},
        extra_checks={"basic_sanity": sanity},
    )

    print(json.dumps({"dataset": DATASET, "protocol": args.protocol, "rows": len(df), "stats": stats, "sanity": sanity}, indent=2))


if __name__ == "__main__":
    main()
