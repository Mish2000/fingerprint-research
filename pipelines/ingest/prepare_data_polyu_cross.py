from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from pipelines.ingest.pair_bundle_utils import (
    build_pairs_split_build_meta,
    build_split_subjects_metadata,
    validate_canonical_pairs_df,
    validate_pairs_split_build_meta,
    validate_split_subjects_metadata,
    write_json,
)

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DATASET = "polyu_cross"
DEFAULT_SEED = 42
DEFAULT_NEG_PER_POS = 3
DEFAULT_FINGER_COL = "frgp"
DEFAULT_POSITIVE_POLICY = "same_subject_same_finger_contactless_to_contact_based"
DEFAULT_NEGATIVE_POLICY = "same_finger_other_subject_same_split"


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def infer_session(path: Path) -> int:
    for part in path.parts:
        m = re.search(r"(?:session|sess|s)(\d+)", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 1

def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _iter_candidate_roots(raw_root: Path) -> List[Path]:
    roots = [
        raw_root,
        raw_root / 'PolyU_Hong_Kong',
        raw_root / 'Cross_Fingerprint_Images_Database',
        repo_root_from_here() / 'data' / 'raw' / 'PolyU_Hong_Kong',
        repo_root_from_here() / 'data' / 'raw' / 'PolyU_Hong_Kong' / 'Cross_Fingerprint_Images_Database',
    ]
    out: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _find_dir_by_name(roots: Iterable[Path], names: Iterable[str]) -> Optional[Path]:
    names_l = {name.lower() for name in names}
    for root in roots:
        if not root.exists():
            continue
        if root.is_dir() and root.name.lower() in names_l:
            return root.resolve()
        for path in root.rglob('*'):
            if path.is_dir() and path.name.lower() in names_l:
                return path.resolve()
    return None


def resolve_polyu_cross_dirs(
    raw_root: Path,
    contactless_raw_dir: Optional[str],
    contactless_processed_dir: Optional[str],
    contact_based_dir: Optional[str],
) -> tuple[Path, Path, Path]:
    roots = _iter_candidate_roots(raw_root)

    def _explicit(path_str: Optional[str]) -> Optional[Path]:
        if not path_str:
            return None
        path = Path(path_str).expanduser().resolve()
        return path if path.exists() else None

    contactless_raw = _explicit(contactless_raw_dir)
    processed = _explicit(contactless_processed_dir)
    contact_based = _explicit(contact_based_dir)

    if contactless_raw is None:
        contactless_raw = _find_dir_by_name(roots, ['contactless_2d_fingerprint_images', 'contactless'])
    if processed is None:
        processed = _find_dir_by_name(roots, ['processed_contactless_2d_fingerprint_images', 'processed_contactless', 'processed'])
    if contact_based is None:
        contact_based = _find_dir_by_name(roots, ['contact-based_fingerprints', 'contact_based', 'contact-based'])

    if contactless_raw is None or contact_based is None:
        raise FileNotFoundError(
            f'Could not resolve PolyU cross directories under {raw_root}. ' 
            'Expected contactless_2d_fingerprint_images and contact-based_fingerprints.'
        )
    if processed is None:
        processed = repo_root_from_here() / 'data' / 'processed' / DATASET

    return contactless_raw, processed, contact_based


def parse_last_int(text: str) -> Optional[int]:
    nums = re.findall(r"\d+", text)
    return int(nums[-1]) if nums else None


def parse_contactless_path(p: Path) -> Optional[dict]:
    subject_id = None
    for part in p.parts:
        m = re.fullmatch(r"p(\d+)", part, flags=re.IGNORECASE)
        if m:
            subject_id = int(m.group(1))

    if subject_id is None:
        return None

    sample_id = parse_last_int(p.stem)

    return {
        "dataset": DATASET,
        "capture": "contactless",
        "subject_id": int(subject_id),
        "impression": f"sample_{int(sample_id):02d}" if sample_id is not None else "sample_unknown",
        "ppi": 0,
        "frgp": 0,
        "path": str(p.resolve()),
        "split": None,
        "sample_id": sample_id,
        "session": infer_session(p),
        "source_modality": "contactless",
    }


def parse_contact_based_path(p: Path) -> Optional[dict]:
    m = re.fullmatch(r"(?P<subject>\d+)[_-](?P<sample>\d+)", p.stem)
    if m is None:
        m = re.search(r"(?P<subject>\d+)[_-](?P<sample>\d+)", p.stem)
    if m is None:
        return None

    subject_id = int(m.group("subject"))
    sample_id = int(m.group("sample"))

    return {
        "dataset": DATASET,
        "capture": "contact_based",
        "subject_id": int(subject_id),
        "impression": f"sample_{int(sample_id):02d}",
        "ppi": 0,
        "frgp": 0,
        "path": str(p.resolve()),
        "split": None,
        "sample_id": sample_id,
        "session": infer_session(p),
        "source_modality": "contact_based",
    }


def choose_contactless_dir(raw_dir: Path, processed_dir: Path, mode: str) -> Path:
    if mode == "processed":
        return processed_dir
    if mode == "raw":
        return raw_dir

    processed_has_images = any(True for _ in iter_images(processed_dir)) if processed_dir.exists() else False
    if processed_has_images:
        return processed_dir
    return raw_dir


def build_manifest(contactless_dir: Path, contact_based_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []

    for p in iter_images(contactless_dir):
        rec = parse_contactless_path(p)
        if rec is not None:
            rows.append(rec)

    for p in iter_images(contact_based_dir):
        rec = parse_contact_based_path(p)
        if rec is not None:
            rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["dataset", "capture", "subject_id", "impression", "ppi", "frgp", "path", "split", "sample_id", "session", "source_modality"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["subject_id", "capture", "sample_id", "path"]).reset_index(drop=True)
    return df


def split_by_subject(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[int]]:
    subjects = sorted(int(s) for s in df["subject_id"].dropna().unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1 if n >= 3 else 0), n)
    n_val = min(max(n_val, 1 if n >= 3 else 0), max(0, n - n_train))
    n_test = max(0, n - n_train - n_val)

    train = subjects[:n_train]
    val = subjects[n_train:n_train + n_val]
    test = subjects[n_train + n_val:n_train + n_val + n_test]

    if not test and val:
        test = [val.pop()]
    if not val and train:
        val = [train.pop()]
    if not train and test:
        train = [test.pop()]

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


def assign_split(df: pd.DataFrame, split_map: Dict[str, List[int]]) -> pd.DataFrame:
    sid_to_split = {}
    for sp, ids in split_map.items():
        for sid in ids:
            sid_to_split[int(sid)] = sp
    out = df.copy()
    out["split"] = out["subject_id"].map(lambda x: sid_to_split.get(int(x), None))
    return out


def choose_one(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["subject_id", "capture", "sample_id", "path"]).reset_index(drop=True)
    return out


def make_positive_pairs(df: pd.DataFrame, max_pos_per_subject: int) -> pd.DataFrame:
    rows: List[dict] = []
    for sid, g in df.groupby("subject_id"):
        cl = g[g["capture"] == "contactless"].sort_values(["sample_id", "path"])
        cb = g[g["capture"] == "contact_based"].sort_values(["sample_id", "path"])
        if cl.empty or cb.empty:
            continue
        combos: List[Tuple[str, str]] = []
        for path_a in cl["path"].tolist():
            for path_b in cb["path"].tolist():
                combos.append((path_a, path_b))
        combos = combos[:max_pos_per_subject]
        split = str(g["split"].iloc[0])
        for path_a, path_b in combos:
            rows.append(
                {
                    "path_a": path_a,
                    "path_b": path_b,
                    "label": 1,
                    "subject_a": int(sid),
                    "subject_b": int(sid),
                    "frgp": 0,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def make_negative_pairs(df: pd.DataFrame, pos: pd.DataFrame, seed: int, neg_per_pos: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[dict] = []
    contact_based_by_split_subject: Dict[Tuple[str, int], List[str]] = {}
    for (split, sid), g in df[df["capture"] == "contact_based"].groupby(["split", "subject_id"]):
        contact_based_by_split_subject[(str(split), int(sid))] = g.sort_values(["sample_id", "path"])["path"].tolist()

    for _, pr in pos.iterrows():
        split = str(pr["split"])
        sid = int(pr["subject_a"])
        candidates = [
            (other_sid, paths)
            for (sp, other_sid), paths in contact_based_by_split_subject.items()
            if sp == split and other_sid != sid and paths
        ]
        if not candidates:
            continue
        for _ in range(int(neg_per_pos)):
            other_sid, paths = rng.choice(candidates)
            rows.append(
                {
                    "path_a": str(pr["path_a"]),
                    "path_b": rng.choice(paths),
                    "label": 0,
                    "subject_a": sid,
                    "subject_b": int(other_sid),
                    "frgp": 0,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def build_split_pairs(pos: pd.DataFrame, neg: pd.DataFrame, split: str) -> pd.DataFrame:
    parts = []
    if not pos.empty:
        parts.append(pos[pos["split"] == split])
    if not neg.empty:
        parts.append(neg[neg["split"] == split])
    if not parts:
        return pd.DataFrame(columns=["pair_id", "label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"])

    df = pd.concat(parts, ignore_index=True)
    df = df.reset_index(drop=True)
    df["pair_id"] = range(len(df))
    return df[["pair_id", "label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"]]


def write_nested_pairs_bundle(
    out_dir: Path,
    split_map: Dict[str, List[int]],
    *,
    seed: int,
    neg_per_pos: int,
    manifest_path: Path,
) -> None:
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for sp in ("train", "val", "test"):
        src = out_dir / f"pairs_{sp}.csv"
        if src.exists():
            shutil.copy2(src, pairs_dir / src.name)

    split_meta = build_split_subjects_metadata(
        splits=split_map,
        seed=seed,
        neg_per_pos=neg_per_pos,
        impostors_per_pos=neg_per_pos,
        same_finger_policy=True,
        negative_pair_policy=DEFAULT_NEGATIVE_POLICY,
        positive_pair_policy=DEFAULT_POSITIVE_POLICY,
        finger_col=DEFAULT_FINGER_COL,
        resolved_data_dir=out_dir,
        manifest_path=manifest_path,
        pair_mode="cross_capture_probe_to_gallery",
        max_pos_per_subject=12,
    )
    validate_split_subjects_metadata(split_meta, context=f"{DATASET} split_subjects metadata")
    write_json(pairs_dir / "split_subjects.json", split_meta)


def sanity_checks(df: pd.DataFrame, split_map: Dict[str, List[int]], pos: pd.DataFrame, neg: pd.DataFrame) -> Dict[str, object]:
    s_train, s_val, s_test = map(set, (split_map["train"], split_map["val"], split_map["test"]))
    disjoint_ok = (len(s_train & s_val) == 0) and (len(s_train & s_test) == 0) and (len(s_val & s_test) == 0)

    leak_rows = 0
    for sp in ("train", "val", "test"):
        ids = set(split_map[sp])
        leak_rows += int(((df["split"] == sp) & (~df["subject_id"].isin(ids))).sum())

    pos_bad = int((pos["subject_a"] != pos["subject_b"]).sum())
    neg_bad = int((neg["subject_a"] == neg["subject_b"]).sum())

    return {
        "disjoint_subject_splits": bool(disjoint_ok),
        "leak_rows": int(leak_rows),
        "positive_subject_mismatch": int(pos_bad),
        "negative_same_subject": int(neg_bad),
        "ok": bool(disjoint_ok and leak_rows == 0 and pos_bad == 0 and neg_bad == 0),
    }


def main() -> None:
    rr = repo_root_from_here()

    ap = argparse.ArgumentParser(description="Prepare PolyU cross-sensor manifest/splits/pairs bundle under data/manifests.")
    ap.add_argument(
        "--raw_root",
        type=str,
        default=str(rr / "data" / "raw" / "PolyU_Hong_Kong"),
    )
    ap.add_argument(
        "--contactless_raw_dir",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--contactless_processed_dir",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--contact_based_dir",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(rr / "data" / "manifests" / DATASET),
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--neg_per_pos", type=int, default=DEFAULT_NEG_PER_POS)
    ap.add_argument("--max_pos_per_subject", type=int, default=12)
    ap.add_argument("--contactless_mode", choices=["auto", "processed", "raw"], default="auto")
    args = ap.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0, 1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val_ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    raw_root = Path(args.raw_root).expanduser().resolve()
    contactless_raw_dir, contactless_processed_dir, contact_based_dir = resolve_polyu_cross_dirs(
        raw_root,
        args.contactless_raw_dir,
        args.contactless_processed_dir,
        args.contact_based_dir,
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    contactless_dir = choose_contactless_dir(contactless_raw_dir, contactless_processed_dir, args.contactless_mode)

    print("Repo root           :", rr)
    print("Contactless dir     :", contactless_dir)
    print("Contact-based dir   :", contact_based_dir)
    print("Out dir             :", out_dir)

    df = build_manifest(contactless_dir, contact_based_dir)
    print("Parsed rows:", len(df))
    if len(df) == 0:
        raise RuntimeError("No PolyU cross rows parsed. Check input directories.")

    split = split_by_subject(df, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    df = assign_split(df, split)
    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    df_one = choose_one(df)
    pos = make_positive_pairs(df_one, max_pos_per_subject=args.max_pos_per_subject)
    neg = make_negative_pairs(df_one, pos, seed=args.seed, neg_per_pos=args.neg_per_pos)

    pos.to_csv(out_dir / "pairs_pos.csv", index=False)
    neg.to_csv(out_dir / "pairs_neg.csv", index=False)

    for sp in ("train", "val", "test"):
        pairs_sp = build_split_pairs(pos, neg, sp)
        pairs_sp = validate_canonical_pairs_df(
            pairs_sp,
            context=f"{DATASET}/{sp} canonical pairs",
            expected_split=sp,
            require_exact_columns=True,
            require_non_empty=True,
        )
        pairs_sp.to_csv(out_dir / f"pairs_{sp}.csv", index=False)

    write_nested_pairs_bundle(
        out_dir,
        split,
        seed=int(args.seed),
        neg_per_pos=int(args.neg_per_pos),
        manifest_path=manifest_path,
    )

    meta = build_pairs_split_build_meta(
        dataset=DATASET,
        seed=int(args.seed),
        neg_per_pos=int(args.neg_per_pos),
        impostors_per_pos=int(args.neg_per_pos),
        finger_col=DEFAULT_FINGER_COL,
        positive_pair_policy=DEFAULT_POSITIVE_POLICY,
        negative_pair_policy=DEFAULT_NEGATIVE_POLICY,
        extra={
            "contactless_dir": str(contactless_dir),
            "contact_based_dir": str(contact_based_dir),
            "out_dir": str(out_dir),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "contactless_mode": args.contactless_mode,
            "pair_mode": "cross_capture_probe_to_gallery",
            "max_pos_per_subject": int(args.max_pos_per_subject),
            "notes": "Contactless probe is matched against contact-based gallery within subject for positives and across subject within split for negatives.",
        },
    )
    validate_pairs_split_build_meta(meta, context=f"{DATASET} pairs_split_build metadata")
    write_json(out_dir / "pairs_split_build.meta.json", meta)

    stats = {
        "manifest_rows": int(len(df)),
        "unique_subjects": int(df["subject_id"].nunique()),
        "contactless_rows": int((df["capture"] == "contactless").sum()),
        "contact_based_rows": int((df["capture"] == "contact_based").sum()),
        "pos_pairs": int(len(pos)),
        "neg_pairs": int(len(neg)),
        "pos_by_split": pos["split"].value_counts().to_dict(),
        "neg_by_split": neg["split"].value_counts().to_dict(),
        "pairs_by_split": {
            sp: int(pd.read_csv(out_dir / f"pairs_{sp}.csv").shape[0])
            for sp in ("train", "val", "test")
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    sanity = sanity_checks(df, split, pos, neg)
    (out_dir / "sanity_report.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    print("\nDONE.")
    print("Stats:\n", json.dumps(stats, indent=2))
    print("Sanity:\n", json.dumps(sanity, indent=2))
    print("Wrote nested pairs bundle to:", out_dir / "pairs")


if __name__ == "__main__":
    main()
