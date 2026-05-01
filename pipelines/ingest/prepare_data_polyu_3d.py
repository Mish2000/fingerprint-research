from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

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
    ensure_protocol_note,
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

DATASET = "polyu_3d"
DEFAULT_FINGER_COL = "frgp"
PROTOCOLS = {
    "surface_only": {
        "positive": "same_subject_within_contactless_3d_surface",
        "negative": "other_subject_within_contactless_3d_surface_same_split",
        "pair_mode": "within_modality_verification",
        "modalities": {"contactless_3d_surface"},
        "same_finger_policy": False,
    },
    "raw_2d_only": {
        "positive": "same_subject_within_contactless_2d_photometric_raw",
        "negative": "other_subject_within_contactless_2d_photometric_raw_same_split",
        "pair_mode": "within_modality_verification",
        "modalities": {"contactless_2d_photometric_raw"},
        "same_finger_policy": False,
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
    "frame_id",
    "variant_rank",
]
SURFACE_RE = re.compile(r"^(?P<subject>\d+)[_-](?P<sample>\d+)$")
P_PART_RE = re.compile(r"^p(?P<value>\d+)$", re.IGNORECASE)
RAW_FRAME_RE = re.compile(r"^(?P<base>.+?)_(?P<frame>[01])$")


def _resolve_dir(path: Optional[str], fallback_candidates: Iterable[Path]) -> Optional[Path]:
    if path:
        p = Path(path).expanduser().resolve()
        return p if p.exists() else p
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _iter_candidate_roots(raw_root: Path) -> Iterator[Path]:
    seen: set[str] = set()
    for candidate in [
        raw_root,
        raw_root / "PolyU_Hong_Kong",
        raw_root / "3D_Fingerprint_Images_Database_V2",
        raw_root / "3D_Fingerprint_Database",
        repo_root_from_here() / "data" / "raw" / "PolyU_Hong_Kong",
        repo_root_from_here() / "data" / "raw" / "PolyU_Hong_Kong" / "3D_Fingerprint_Images_Database_V2",
        repo_root_from_here() / "data" / "raw" / "PolyU_Hong_Kong" / "3D_Fingerprint_Database",
    ]:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def _find_dir_by_names(roots: Iterable[Path], target_names: Iterable[str]) -> Optional[Path]:
    target_names_l = {name.lower() for name in target_names}
    for root in roots:
        if not root.exists():
            continue
        if root.is_dir() and root.name.lower() in target_names_l:
            return root.resolve()
        for path in root.rglob('*'):
            if path.is_dir() and path.name.lower() in target_names_l:
                return path.resolve()
    return None


def resolve_polyu_dirs(raw_root: Path, raw_2d_dir: Optional[str], surface_dir: Optional[str]) -> tuple[Path, Path]:
    roots = list(_iter_candidate_roots(raw_root))

    raw_2d = _resolve_dir(raw_2d_dir, [])
    surface = _resolve_dir(surface_dir, [])

    if raw_2d is None:
        raw_2d = _find_dir_by_names(
            roots,
            ["3D_fingerprint_image", "3D_fingerpring_image", "2D_fingerprint_images"],
        )
    if surface is None:
        surface = _find_dir_by_names(
            roots,
            ["3D_surface", "3D_FingerPrint_Database", "surface", "surfaces"],
        )

    if raw_2d is None or surface is None:
        raise FileNotFoundError(
            f"Could not resolve PolyU 3D directories under {raw_root}. "
            "Expected names like 3D_fingerprint_image and 3D_surface under PolyU_Hong_Kong."
        )
    return raw_2d, surface


def _parse_subject_from_parts(path: Path) -> Optional[int]:
    for part in reversed(path.parts[:-1]):
        m = P_PART_RE.match(part)
        if m:
            return int(m.group("value"))
    return None


def _group_raw_impressions(raw_2d_dir: Path) -> Dict[tuple[int, int], Dict[str, int]]:
    groups: Dict[tuple[int, int], List[str]] = {}
    for path in iter_images(raw_2d_dir):
        subject_id = _parse_subject_from_parts(path)
        if subject_id is None:
            continue
        session = infer_session(path)
        m = RAW_FRAME_RE.match(path.stem)
        base = m.group("base") if m else path.stem
        groups.setdefault((subject_id, session), []).append(base)
    out: Dict[tuple[int, int], Dict[str, int]] = {}
    for key, bases in groups.items():
        ordered = sorted(set(bases))
        out[key] = {base: idx + 1 for idx, base in enumerate(ordered)}
    return out


def build_manifest(raw_2d_dir: Path, surface_dir: Path) -> tuple[pd.DataFrame, int]:
    rows: List[dict] = []
    calibration_files = sum(1 for p in surface_dir.glob("pixelOrients_*.txt") if p.is_file())
    raw_impression_map = _group_raw_impressions(raw_2d_dir)

    for path in sorted(raw_2d_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        subject_id = _parse_subject_from_parts(path)
        if subject_id is None:
            continue
        session = infer_session(path)
        m = RAW_FRAME_RE.match(path.stem)
        base = m.group("base") if m else path.stem
        frame_id = int(m.group("frame")) if m else 0
        sample_id = raw_impression_map.get((subject_id, session), {}).get(base, 0)
        rows.append(
            {
                "dataset": DATASET,
                "capture": "contactless",
                "subject_id": int(subject_id),
                "impression": f"sample_{int(sample_id):02d}",
                "ppi": 0,
                "frgp": 0,
                "path": str(path.resolve()),
                "split": None,
                "sample_id": int(sample_id),
                "session": int(session),
                "source_modality": "contactless_2d_photometric_raw",
                "frame_id": int(frame_id),
                "variant_rank": 0,
            }
        )

    for path in iter_files(surface_dir, exts=[".mat"]):
        m = SURFACE_RE.match(path.stem)
        if not m:
            continue
        rows.append(
            {
                "dataset": DATASET,
                "capture": "contactless",
                "subject_id": int(m.group("subject")),
                "impression": f"sample_{int(m.group('sample')):02d}",
                "ppi": 0,
                "frgp": 0,
                "path": str(path.resolve()),
                "split": None,
                "sample_id": int(m.group("sample")),
                "session": int(infer_session(path)),
                "source_modality": "contactless_3d_surface",
                "frame_id": 0,
                "variant_rank": 0,
            }
        )

    if not rows:
        return pd.DataFrame(columns=IMAGE_COLUMNS), calibration_files

    df = pd.DataFrame(rows)
    df = df.sort_values(["subject_id", "session", "source_modality", "sample_id", "frame_id", "path"]).reset_index(drop=True)
    return df, calibration_files


def _canonical_pairs(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["pair_id", "label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"])
    return pd.DataFrame(rows, columns=["label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"])


def make_within_modality_pairs(df: pd.DataFrame, *, modalities: set[str], max_pos_per_subject: int, seed: int, neg_per_pos: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = choose_one_per_key(
        df[df["source_modality"].isin(modalities)],
        ["subject_id", "impression", "source_modality", "split"],
        sort_cols=["subject_id", "source_modality", "sample_id", "frame_id", "path"],
    )

    pos_rows: List[dict] = []
    for (split_name, subject_id, source_modality), group in base.groupby(["split", "subject_id", "source_modality"]):
        paths = group.sort_values(["sample_id", "path"])["path"].tolist()
        combos: List[tuple[str, str]] = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                combos.append((paths[i], paths[j]))
        for path_a, path_b in combos[:max_pos_per_subject]:
            pos_rows.append(
                {
                    "label": 1,
                    "split": str(split_name),
                    "subject_a": int(subject_id),
                    "subject_b": int(subject_id),
                    "frgp": 0,
                    "path_a": path_a,
                    "path_b": path_b,
                }
            )
    pos = _canonical_pairs(pos_rows)

    rng = random.Random(seed)
    pool: Dict[tuple[str, str], List[tuple[int, str]]] = {}
    path_to_modality = {}
    for (split_name, source_modality), group in base.groupby(["split", "source_modality"]):
        pool[(str(split_name), str(source_modality))] = [(int(row.subject_id), str(row.path)) for row in group[["subject_id", "path"]].itertuples(index=False)]
        for row in group[["path", "source_modality"]].itertuples(index=False):
            path_to_modality[str(row.path)] = str(row.source_modality)

    neg_rows: List[dict] = []
    for row in pos.itertuples(index=False):
        source_modality = path_to_modality.get(str(row.path_a))
        candidates = [item for item in pool.get((str(row.split), str(source_modality)), []) if int(item[0]) != int(row.subject_a)]
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
                    "frgp": 0,
                    "path_a": str(row.path_a),
                    "path_b": str(path_b),
                }
            )
    return pos, _canonical_pairs(neg_rows)


def main() -> None:
    rr = repo_root_from_here()
    ap = argparse.ArgumentParser(description="Prepare optional PolyU 3D manifest + pair bundle under data/manifests.")
    ap.add_argument("--raw_root", type=str, default=str(rr / "data" / "raw" / "PolyU_Hong_Kong"))
    ap.add_argument("--raw_2d_dir", type=str, default=None)
    ap.add_argument("--surface_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=str(rr / "data" / "manifests" / DATASET))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--neg_per_pos", type=int, default=DEFAULT_NEG_PER_POS)
    ap.add_argument("--max_pos_per_subject", type=int, default=10)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS.keys()), default="surface_only")
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    raw_2d_dir, surface_dir = resolve_polyu_dirs(raw_root, args.raw_2d_dir, args.surface_dir)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_protocol_note(DATASET, out_dir)

    df, calibration_files = build_manifest(raw_2d_dir, surface_dir)
    if df.empty:
        raise RuntimeError("No PolyU 3D rows parsed. Check input directories.")

    split_map = split_by_subject(df, args.seed, args.train_ratio, args.val_ratio)
    write_split_json(out_dir, split_map)
    df = assign_split(df, split_map)
    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    protocol = PROTOCOLS[args.protocol]
    pos, neg = make_within_modality_pairs(
        df,
        modalities=set(protocol["modalities"]),
        max_pos_per_subject=int(args.max_pos_per_subject),
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
        max_pos_per_subject=int(args.max_pos_per_subject),
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
            "raw_2d_dir": str(raw_2d_dir),
            "surface_dir": str(surface_dir),
            "protocol": str(args.protocol),
            "pair_mode": str(protocol["pair_mode"]),
            "max_pos_per_subject": int(args.max_pos_per_subject),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "notes": "PolyU 3D optional ingest preserves both raw photometric 2D acquisition frames and reconstructed 3D surfaces in the manifest. The canonical pair bundle stays within-modality because the raw frames are reconstruction inputs, not a stable peer biometric modality.",
        },
    )
    stats = stats_with_pairs(
        df,
        pos,
        neg,
        extra={
            "protocol": str(args.protocol),
            "pair_eligible_source_modalities": sorted(list(protocol["modalities"])),
            "raw_photometric_rows": int((df["source_modality"] == "contactless_2d_photometric_raw").sum()),
            "surface_rows": int((df["source_modality"] == "contactless_3d_surface").sum()),
            "calibration_sidecar_files": int(calibration_files),
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
        extra_manifest={"protocol": str(args.protocol), "pair_eligible_source_modalities": sorted(list(protocol["modalities"]))},
        extra_checks={"basic_sanity": sanity},
    )

    print(json.dumps({"dataset": DATASET, "rows": len(df), "stats": stats, "sanity": sanity}, indent=2))


if __name__ == "__main__":
    main()
