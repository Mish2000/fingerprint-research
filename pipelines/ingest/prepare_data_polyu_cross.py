from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Avoid importing broken optional pandas accelerators in mixed NumPy environments.
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

from pipelines.ingest.pair_bundle_utils import write_json

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DATASET = "polyu_cross"
DEFAULT_SEED = 42
DEFAULT_NEG_PER_POS = 3
DEFAULT_FINGER_COL = "finger_unit_id"
DEFAULT_POSITIVE_POLICY = "same_subject_same_finger_unit_contactless_to_contact_based"
DEFAULT_NEGATIVE_POLICY = "different_subject_finger_unit_same_split_contact_based_without_replacement"
PAIR_SCHEMA_VERSION = "v3_polyu_cross_pair_csv_phase2a"
PAIR_BUILD_SCHEMA_VERSION = "v3_polyu_cross_pair_bundle_phase2a"
SPLIT_SUBJECTS_SCHEMA_VERSION = "v3_polyu_cross_split_subjects_phase2a"
MANIFEST_COLUMNS = [
    "dataset",
    "capture",
    "subject_id",
    "finger_unit_id",
    "impression",
    "ppi",
    "frgp",
    "path",
    "split",
    "sample_id",
    "capture_id",
    "sample_uid",
    "session",
    "session_id",
    "source_modality",
    "modality",
]
PAIR_COLUMNS = [
    "pair_id",
    "label",
    "split",
    "subject_a",
    "subject_b",
    "finger_unit_a",
    "finger_unit_b",
    "frgp",
    "path_a",
    "path_b",
    "modality_a",
    "modality_b",
    "session_a",
    "session_b",
    "sample_uid_a",
    "sample_uid_b",
]
SESSION_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def infer_session_id(path: Path) -> str:
    for part in path.parts:
        m = re.search(r"(?:session|sess)[^\d]*(\d+)", part, flags=re.IGNORECASE)
        if m:
            return _session_label(int(m.group(1)))
        m = re.search(r"(\d+)(?:st|nd|rd|th)?[^\w]*(?:session|sess)", part, flags=re.IGNORECASE)
        if m:
            return _session_label(int(m.group(1)))
        m = re.fullmatch(r"s(\d+)", part, flags=re.IGNORECASE)
        if m:
            return _session_label(int(m.group(1)))
        tokens = [token for token in re.split(r"[^a-z0-9]+", part.lower()) if token]
        if "session" in tokens or "sess" in tokens:
            for token in tokens:
                if token in SESSION_ORDINALS:
                    return _session_label(SESSION_ORDINALS[token])
                m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)?", token)
                if m:
                    return _session_label(int(m.group(1)))
    return "unknown"


def infer_session(path: Path) -> int:
    session_id = infer_session_id(path)
    if session_id == "session_1":
        return 1
    if session_id == "session_2":
        return 2
    return 0


def _session_label(value: int) -> str:
    if int(value) == 1:
        return "session_1"
    if int(value) == 2:
        return "session_2"
    return "unknown"


def infer_path_base(*roots: Path) -> Path:
    existing = [Path(root).expanduser().resolve() for root in roots]
    if not existing:
        return repo_root_from_here()
    common = os.path.commonpath([str(root) for root in existing])
    return Path(common)


def format_path(path: Path, path_base: Optional[Path]) -> str:
    resolved = path.resolve()
    if path_base is None:
        return str(resolved)
    try:
        return resolved.relative_to(path_base.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def make_sample_uid(
    *,
    capture: str,
    subject_id: int,
    finger_unit_id: int,
    sample_id: Optional[int],
    session_id: str,
    path: str,
) -> str:
    key = f"{DATASET}|{capture}|{subject_id}|{finger_unit_id}|{sample_id}|{session_id}|{path}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{DATASET}_{digest}"


def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _iter_candidate_roots(raw_root: Path) -> List[Path]:
    roots = [
        raw_root,
        raw_root / 'PolyU_Hong_Kong',
        raw_root / 'PolyU Hong Kong',
        raw_root / 'Cross_Fingerprint_Images_Database',
        repo_root_from_here() / 'data' / 'raw' / 'PolyU_Hong_Kong',
        repo_root_from_here() / 'data' / 'raw' / 'PolyU_Hong_Kong' / 'Cross_Fingerprint_Images_Database',
        repo_root_from_here() / 'data' / 'raw' / 'PolyU Hong Kong',
        repo_root_from_here() / 'data' / 'raw' / 'PolyU Hong Kong' / 'Cross_Fingerprint_Images_Database',
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
        contactless_raw = _find_dir_by_name(roots, ['contactless_2d_fingerprint_images', 'contactless 2d fingerprint images', 'contactless'])
    if processed is None:
        processed = _find_dir_by_name(roots, ['processed_contactless_2d_fingerprint_images', 'processed contactless 2d fingerprint images', 'processed_contactless', 'processed'])
    if contact_based is None:
        contact_based = _find_dir_by_name(roots, ['contact-based_fingerprints', 'contact-based fingerprints', 'contact_based', 'contact-based'])

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


def parse_contactless_path(p: Path, *, path_base: Optional[Path] = None) -> Optional[dict]:
    subject_id = None
    for part in p.parts:
        m = re.fullmatch(r"p(\d+)", part, flags=re.IGNORECASE)
        if m:
            subject_id = int(m.group(1))

    if subject_id is None:
        return None

    sample_id = parse_last_int(p.stem)
    finger_unit_id = int(subject_id)
    capture_id = f"sample_{int(sample_id):02d}" if sample_id is not None else "sample_unknown"
    session_id = infer_session_id(p)
    manifest_path = format_path(p, path_base)

    return {
        "dataset": DATASET,
        "capture": "contactless",
        "subject_id": int(subject_id),
        "finger_unit_id": int(finger_unit_id),
        "impression": capture_id,
        "ppi": 0,
        "frgp": 0,
        "path": manifest_path,
        "split": None,
        "sample_id": sample_id,
        "capture_id": capture_id,
        "sample_uid": make_sample_uid(
            capture="contactless",
            subject_id=int(subject_id),
            finger_unit_id=int(finger_unit_id),
            sample_id=sample_id,
            session_id=session_id,
            path=manifest_path,
        ),
        "session": infer_session(p),
        "session_id": session_id,
        "source_modality": "contactless",
        "modality": "contactless_2d",
    }


def parse_contact_based_path(p: Path, *, path_base: Optional[Path] = None) -> Optional[dict]:
    m = re.fullmatch(r"(?P<subject>\d+)[_-](?P<sample>\d+)", p.stem)
    if m is None:
        m = re.search(r"(?P<subject>\d+)[_-](?P<sample>\d+)", p.stem)
    if m is None:
        return None

    subject_id = int(m.group("subject"))
    sample_id = int(m.group("sample"))
    finger_unit_id = int(subject_id)
    capture_id = f"sample_{int(sample_id):02d}"
    session_id = infer_session_id(p)
    manifest_path = format_path(p, path_base)

    return {
        "dataset": DATASET,
        "capture": "contact_based",
        "subject_id": int(subject_id),
        "finger_unit_id": int(finger_unit_id),
        "impression": capture_id,
        "ppi": 0,
        "frgp": 0,
        "path": manifest_path,
        "split": None,
        "sample_id": sample_id,
        "capture_id": capture_id,
        "sample_uid": make_sample_uid(
            capture="contact_based",
            subject_id=int(subject_id),
            finger_unit_id=int(finger_unit_id),
            sample_id=sample_id,
            session_id=session_id,
            path=manifest_path,
        ),
        "session": infer_session(p),
        "session_id": session_id,
        "source_modality": "contact_based",
        "modality": "contact_based_2d",
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


def build_manifest(
    contactless_dir: Path,
    contact_based_dir: Path,
    *,
    path_base: Optional[Path] = None,
) -> pd.DataFrame:
    if path_base is None:
        path_base = infer_path_base(contactless_dir, contact_based_dir)
    rows: List[dict] = []

    for p in iter_images(contactless_dir):
        rec = parse_contactless_path(p, path_base=path_base)
        if rec is not None:
            rows.append(rec)

    for p in iter_images(contact_based_dir):
        rec = parse_contact_based_path(p, path_base=path_base)
        if rec is not None:
            rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    df = pd.DataFrame(rows)
    df = df[MANIFEST_COLUMNS].sort_values(
        ["subject_id", "finger_unit_id", "capture", "session", "sample_id", "path"],
        kind="mergesort",
    ).reset_index(drop=True)
    if not df["sample_uid"].is_unique:
        duplicates = df[df["sample_uid"].duplicated()]["sample_uid"].head(5).tolist()
        raise RuntimeError(f"PolyU Cross sample_uid values must be globally unique. Examples: {duplicates}")
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
    out = out.sort_values(["subject_id", "finger_unit_id", "capture", "session", "sample_id", "path"]).reset_index(drop=True)
    return out


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=PAIR_COLUMNS)


def _select_positive_gallery(contactless_row: pd.Series, contact_based: pd.DataFrame) -> Optional[pd.Series]:
    ordered = contact_based.sort_values(["session", "sample_id", "path"], kind="mergesort")
    priorities = [
        (ordered["session_id"] == contactless_row["session_id"]) & (ordered["capture_id"] == contactless_row["capture_id"]),
        ordered["capture_id"] == contactless_row["capture_id"],
        ordered["session_id"] == contactless_row["session_id"],
    ]
    for mask in priorities:
        candidates = ordered[mask]
        if not candidates.empty:
            return candidates.iloc[0]
    if ordered.empty:
        return None
    return ordered.iloc[0]


def _pair_record(row_a: pd.Series, row_b: pd.Series, *, label: int) -> dict:
    return {
        "label": int(label),
        "split": str(row_a["split"]),
        "subject_a": int(row_a["subject_id"]),
        "subject_b": int(row_b["subject_id"]),
        "finger_unit_a": int(row_a["finger_unit_id"]),
        "finger_unit_b": int(row_b["finger_unit_id"]),
        "frgp": 0,
        "path_a": str(row_a["path"]),
        "path_b": str(row_b["path"]),
        "modality_a": str(row_a["modality"]),
        "modality_b": str(row_b["modality"]),
        "session_a": str(row_a["session_id"]),
        "session_b": str(row_b["session_id"]),
        "sample_uid_a": str(row_a["sample_uid"]),
        "sample_uid_b": str(row_b["sample_uid"]),
    }


def make_positive_pairs(df: pd.DataFrame, max_pos_per_subject: int = 0) -> pd.DataFrame:
    rows: List[dict] = []
    for (_, _, _), g in df.groupby(["split", "subject_id", "finger_unit_id"], dropna=False):
        cl = g[g["capture"] == "contactless"].sort_values(["session", "sample_id", "path"], kind="mergesort")
        cb = g[g["capture"] == "contact_based"].sort_values(["session", "sample_id", "path"], kind="mergesort")
        if cl.empty or cb.empty:
            continue
        if int(max_pos_per_subject) > 0:
            cl = cl.head(int(max_pos_per_subject))
        for _, contactless_row in cl.iterrows():
            gallery_row = _select_positive_gallery(contactless_row, cb)
            if gallery_row is None:
                continue
            rows.append(_pair_record(contactless_row, gallery_row, label=1))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c for c in PAIR_COLUMNS if c != "pair_id"])


def make_negative_pairs(df: pd.DataFrame, pos: pd.DataFrame, seed: int, neg_per_pos: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[dict] = []
    contact_based = df[df["capture"] == "contact_based"].sort_values(
        ["split", "subject_id", "finger_unit_id", "session", "sample_id", "path"],
        kind="mergesort",
    )
    manifest_by_uid = df.set_index("sample_uid", drop=False)

    for _, pr in pos.sort_values(["split", "subject_a", "finger_unit_a", "sample_uid_a", "sample_uid_b"], kind="mergesort").iterrows():
        split = str(pr["split"])
        subject_a = int(pr["subject_a"])
        finger_unit_a = int(pr["finger_unit_a"])
        row_a = manifest_by_uid.loc[str(pr["sample_uid_a"])]
        candidates = contact_based[
            (contact_based["split"] == split)
            & (contact_based["subject_id"].astype(int) != subject_a)
            & (contact_based["finger_unit_id"].astype(int) != finger_unit_a)
        ]
        if candidates.empty:
            continue
        candidate_rows = list(candidates.itertuples(index=False))
        sampled = rng.sample(candidate_rows, k=min(int(neg_per_pos), len(candidate_rows)))
        for candidate in sampled:
            row_b = pd.Series(candidate._asdict())
            rows.append(_pair_record(row_a, row_b, label=0))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c for c in PAIR_COLUMNS if c != "pair_id"])


def finalize_pair_bundle(pos: pd.DataFrame, neg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    if not pos.empty:
        parts.append(pos.copy())
    if not neg.empty:
        parts.append(neg.copy())
    if not parts:
        empty = _empty_pairs()
        return empty.copy(), empty.copy(), empty.copy()

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["label", "split", "sample_uid_a", "sample_uid_b", "path_a", "path_b"],
        keep="first",
    )
    combined = combined.sort_values(
        [
            "split",
            "label",
            "subject_a",
            "subject_b",
            "finger_unit_a",
            "finger_unit_b",
            "session_a",
            "session_b",
            "sample_uid_a",
            "sample_uid_b",
            "path_a",
            "path_b",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    combined.insert(0, "pair_id", range(len(combined)))
    combined = combined[PAIR_COLUMNS]
    validate_polyu_pairs(combined)
    pos_out = combined[combined["label"] == 1].reset_index(drop=True)
    neg_out = combined[combined["label"] == 0].reset_index(drop=True)
    return pos_out, neg_out, combined


def build_split_pairs(pos: pd.DataFrame, neg: pd.DataFrame, split: str) -> pd.DataFrame:
    parts = []
    if not pos.empty:
        parts.append(pos[pos["split"] == split])
    if not neg.empty:
        parts.append(neg[neg["split"] == split])
    if not parts:
        return _empty_pairs()

    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["pair_id"], kind="mergesort").reset_index(drop=True)[PAIR_COLUMNS]


def validate_polyu_pairs(df: pd.DataFrame) -> None:
    if list(df.columns) != PAIR_COLUMNS:
        raise ValueError(f"PolyU Cross pair columns must be {PAIR_COLUMNS}; found {list(df.columns)}")
    if not df["pair_id"].is_unique:
        raise ValueError("PolyU Cross pair_id values must be globally unique")
    if df.duplicated(subset=["label", "split", "sample_uid_a", "sample_uid_b", "path_a", "path_b"]).any():
        raise ValueError("PolyU Cross pair bundle contains duplicate pair rows")
    bad_direction = df[(df["modality_a"] != "contactless_2d") | (df["modality_b"] != "contact_based_2d")]
    if not bad_direction.empty:
        raise ValueError(f"PolyU Cross pairs must be contactless_2d -> contact_based_2d; bad rows={len(bad_direction)}")
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    pos_bad = pos[(pos["subject_a"] != pos["subject_b"]) | (pos["finger_unit_a"] != pos["finger_unit_b"])]
    if not pos_bad.empty:
        raise ValueError(f"PolyU Cross positive pairs must keep same subject/finger_unit; bad rows={len(pos_bad)}")
    neg_bad = neg[(neg["subject_a"] == neg["subject_b"]) | (neg["finger_unit_a"] == neg["finger_unit_b"])]
    if not neg_bad.empty:
        raise ValueError(f"PolyU Cross negative pairs must use different subject/finger_unit; bad rows={len(neg_bad)}")


def write_nested_pairs_bundle(
    out_dir: Path,
    split_map: Dict[str, List[int]],
    *,
    seed: int,
    neg_per_pos: int,
    manifest_path: Path,
    path_base: Path,
    max_pos_per_subject: int,
) -> None:
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for sp in ("train", "val", "test"):
        src = out_dir / f"pairs_{sp}.csv"
        if src.exists():
            shutil.copy2(src, pairs_dir / src.name)

    split_meta = {
        "schema_version": SPLIT_SUBJECTS_SCHEMA_VERSION,
        "seed": int(seed),
        "neg_per_pos": int(neg_per_pos),
        "impostors_per_pos": int(neg_per_pos),
        "same_finger_policy": "not_applicable_frgp_unknown_constant_use_finger_unit_id",
        "negative_pair_policy": DEFAULT_NEGATIVE_POLICY,
        "positive_pair_policy": DEFAULT_POSITIVE_POLICY,
        "finger_col": DEFAULT_FINGER_COL,
        "pair_schema_version": PAIR_SCHEMA_VERSION,
        "pair_columns": list(PAIR_COLUMNS),
        "splits": {split: sorted(int(x) for x in ids) for split, ids in split_map.items()},
        "resolved_data_dir": str(path_base),
        "manifest_path": str(manifest_path),
        "max_pos_per_subject": int(max_pos_per_subject),
        "pair_mode": "cross_modality_contactless_probe_to_contact_based_gallery",
        "identity_semantics": (
            "PolyU X/pX identifiers represent a client/finger unit in the dataset readme. "
            "The manifest exposes that value as finger_unit_id; frgp is unknown and constant 0."
        ),
    }
    write_json(pairs_dir / "split_subjects.json", split_meta)


def sanity_checks(df: pd.DataFrame, split_map: Dict[str, List[int]], pos: pd.DataFrame, neg: pd.DataFrame, all_pairs: pd.DataFrame) -> Dict[str, object]:
    s_train, s_val, s_test = map(set, (split_map["train"], split_map["val"], split_map["test"]))
    disjoint_ok = (len(s_train & s_val) == 0) and (len(s_train & s_test) == 0) and (len(s_val & s_test) == 0)

    leak_rows = 0
    for sp in ("train", "val", "test"):
        ids = set(split_map[sp])
        leak_rows += int(((df["split"] == sp) & (~df["subject_id"].isin(ids))).sum())

    pos_subject_bad = int((pos["subject_a"] != pos["subject_b"]).sum()) if not pos.empty else 0
    pos_finger_unit_bad = int((pos["finger_unit_a"] != pos["finger_unit_b"]).sum()) if not pos.empty else 0
    neg_subject_bad = int((neg["subject_a"] == neg["subject_b"]).sum()) if not neg.empty else 0
    neg_finger_unit_bad = int((neg["finger_unit_a"] == neg["finger_unit_b"]).sum()) if not neg.empty else 0
    duplicate_pair_rows = (
        int(all_pairs.duplicated(subset=["label", "split", "sample_uid_a", "sample_uid_b", "path_a", "path_b"]).sum())
        if not all_pairs.empty
        else 0
    )
    bad_direction = (
        int(((all_pairs["modality_a"] != "contactless_2d") | (all_pairs["modality_b"] != "contact_based_2d")).sum())
        if not all_pairs.empty
        else 0
    )
    pair_id_unique = bool(all_pairs["pair_id"].is_unique) if not all_pairs.empty else True
    sample_uid_unique = bool(df["sample_uid"].is_unique) if "sample_uid" in df.columns else False

    return {
        "disjoint_subject_splits": bool(disjoint_ok),
        "leak_rows": int(leak_rows),
        "sample_uid_unique": bool(sample_uid_unique),
        "pair_id_globally_unique": bool(pair_id_unique),
        "duplicate_pair_rows": int(duplicate_pair_rows),
        "bad_modality_direction_rows": int(bad_direction),
        "positive_subject_mismatch": int(pos_subject_bad),
        "positive_finger_unit_mismatch": int(pos_finger_unit_bad),
        "negative_same_subject": int(neg_subject_bad),
        "negative_same_finger_unit": int(neg_finger_unit_bad),
        "frgp_unknown_constant": bool(set(df["frgp"].dropna().astype(int).unique().tolist()) == {0}),
        "ok": bool(
            disjoint_ok
            and leak_rows == 0
            and sample_uid_unique
            and pair_id_unique
            and duplicate_pair_rows == 0
            and bad_direction == 0
            and pos_subject_bad == 0
            and pos_finger_unit_bad == 0
            and neg_subject_bad == 0
            and neg_finger_unit_bad == 0
        ),
    }


def build_pair_metadata(
    *,
    seed: int,
    neg_per_pos: int,
    contactless_dir: Path,
    contact_based_dir: Path,
    out_dir: Path,
    path_base: Path,
    train_ratio: float,
    val_ratio: float,
    contactless_mode: str,
    max_pos_per_subject: int,
) -> dict:
    return {
        "dataset": DATASET,
        "seed": int(seed),
        "neg_per_pos": int(neg_per_pos),
        "impostors_per_pos": int(neg_per_pos),
        "finger_col": DEFAULT_FINGER_COL,
        "positive_pair_policy": DEFAULT_POSITIVE_POLICY,
        "negative_pair_policy": DEFAULT_NEGATIVE_POLICY,
        "schema_version": PAIR_BUILD_SCHEMA_VERSION,
        "pair_schema_version": PAIR_SCHEMA_VERSION,
        "pair_columns": list(PAIR_COLUMNS),
        "contactless_dir": str(contactless_dir),
        "contact_based_dir": str(contact_based_dir),
        "path_base": str(path_base),
        "path_policy": "CSV paths are written relative to path_base when the source file is below path_base.",
        "out_dir": str(out_dir),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(1.0 - train_ratio - val_ratio),
        "contactless_mode": contactless_mode,
        "pair_mode": "cross_modality_contactless_probe_to_contact_based_gallery",
        "max_pos_per_subject": int(max_pos_per_subject),
        "positive_sampling_policy": (
            "Each selected contactless probe is paired once to a contact-based mate from the same "
            "subject/finger_unit, preferring same session and capture_id. "
            "max_pos_per_subject=0 means no cap; positive caps are applied after deterministic sorting."
        ),
        "negative_sampling_policy": (
            "For each positive, sample up to neg_per_pos contact-based galleries without replacement "
            "from different subject/finger_unit values in the same split."
        ),
        "identity_semantics": (
            "PolyU X/pX identifiers represent a client/finger unit in the dataset readme. "
            "The manifest exposes that value as finger_unit_id and also keeps it in subject_id for "
            "legacy split compatibility."
        ),
        "frgp_semantics": "frgp is unknown for PolyU Cross and remains constant 0; negatives do not claim same finger-position matching.",
    }


def build_stats(
    df: pd.DataFrame,
    pos: pd.DataFrame,
    neg: pd.DataFrame,
    *,
    max_pos_per_subject: int,
    neg_per_pos: int,
) -> dict:
    contactless = df[df["capture"] == "contactless"]
    contactless_used = int(pos["sample_uid_a"].nunique()) if not pos.empty else 0
    return {
        "manifest_rows": int(len(df)),
        "unique_subjects": int(df["subject_id"].nunique()),
        "unique_finger_units": int(df["finger_unit_id"].nunique()),
        "frgp_values": sorted(int(x) for x in df["frgp"].dropna().unique().tolist()),
        "contactless_rows": int((df["capture"] == "contactless").sum()),
        "contact_based_rows": int((df["capture"] == "contact_based").sum()),
        "contactless_probes_available": int(len(contactless)),
        "contactless_probes_used": int(contactless_used),
        "contactless_probe_use_rate": float(contactless_used / len(contactless)) if len(contactless) else 0.0,
        "deliberate_positive_subset": bool(int(max_pos_per_subject) > 0),
        "max_pos_per_subject": int(max_pos_per_subject),
        "neg_per_pos": int(neg_per_pos),
        "pos_pairs": int(len(pos)),
        "neg_pairs": int(len(neg)),
        "pos_by_split": {str(k): int(v) for k, v in pos["split"].value_counts().to_dict().items()} if not pos.empty else {},
        "neg_by_split": {str(k): int(v) for k, v in neg["split"].value_counts().to_dict().items()} if not neg.empty else {},
        "session_counts": {str(k): int(v) for k, v in df["session_id"].value_counts().to_dict().items()},
        "positive_sampling_policy": (
            "one contact-based mate per selected contactless probe; prefer same session and capture_id"
        ),
        "negative_sampling_policy": "without replacement per positive from different subject/finger_unit in same split",
        "identity_semantics": "subject_id is the PolyU client/finger-unit identifier; finger_unit_id exposes that explicitly",
        "frgp_semantics": "unknown/constant 0; pair generation does not use frgp for same-position negatives",
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
    ap.add_argument("--max_pos_per_subject", type=int, default=0, help="Cap selected contactless probes per subject/finger_unit; 0 means use all.")
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
    path_base = infer_path_base(contactless_dir, contact_based_dir)

    print("Repo root           :", rr)
    print("Contactless dir     :", contactless_dir)
    print("Contact-based dir   :", contact_based_dir)
    print("Path base           :", path_base)
    print("Out dir             :", out_dir)

    df = build_manifest(contactless_dir, contact_based_dir, path_base=path_base)
    print("Parsed rows:", len(df))
    if len(df) == 0:
        raise RuntimeError("No PolyU cross rows parsed. Check input directories.")

    split = split_by_subject(df, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    df = assign_split(df, split)
    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    df_one = choose_one(df)
    pos_raw = make_positive_pairs(df_one, max_pos_per_subject=args.max_pos_per_subject)
    neg_raw = make_negative_pairs(df_one, pos_raw, seed=args.seed, neg_per_pos=args.neg_per_pos)
    pos, neg, all_pairs = finalize_pair_bundle(pos_raw, neg_raw)

    pos.to_csv(out_dir / "pairs_pos.csv", index=False)
    neg.to_csv(out_dir / "pairs_neg.csv", index=False)

    for sp in ("train", "val", "test"):
        pairs_sp = build_split_pairs(pos, neg, sp)
        pairs_sp.to_csv(out_dir / f"pairs_{sp}.csv", index=False)

    write_nested_pairs_bundle(
        out_dir,
        split,
        seed=int(args.seed),
        neg_per_pos=int(args.neg_per_pos),
        manifest_path=manifest_path,
        path_base=path_base,
        max_pos_per_subject=int(args.max_pos_per_subject),
    )

    meta = build_pair_metadata(
        seed=int(args.seed),
        neg_per_pos=int(args.neg_per_pos),
        contactless_dir=contactless_dir,
        contact_based_dir=contact_based_dir,
        out_dir=out_dir,
        path_base=path_base,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        contactless_mode=str(args.contactless_mode),
        max_pos_per_subject=int(args.max_pos_per_subject),
    )
    write_json(out_dir / "pairs_split_build.meta.json", meta)

    stats = build_stats(df, pos, neg, max_pos_per_subject=int(args.max_pos_per_subject), neg_per_pos=int(args.neg_per_pos))
    stats["pairs_by_split"] = {
        sp: int(pd.read_csv(out_dir / f"pairs_{sp}.csv").shape[0])
        for sp in ("train", "val", "test")
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    sanity = sanity_checks(df, split, pos, neg, all_pairs)
    (out_dir / "sanity_report.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    print("\nDONE.")
    print("Stats:\n", json.dumps(stats, indent=2))
    print("Sanity:\n", json.dumps(sanity, indent=2))
    print("Wrote nested pairs bundle to:", out_dir / "pairs")


if __name__ == "__main__":
    main()
