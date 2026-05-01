from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from pipelines.ingest.pair_bundle_utils import (
    CANONICAL_PAIR_COLUMNS,
    build_pairs_split_build_meta,
    build_split_subjects_metadata,
    canonicalize_pairs_df,
    validate_canonical_pairs_df,
    validate_pairs_split_build_meta,
    validate_split_subjects_metadata,
    write_json,
)

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_SEED = 42
DEFAULT_NEG_PER_POS = 3
DEFAULT_SPLITS = ("train", "val", "test")


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_protocol_note(dataset: str, out_dir: Path) -> None:
    src = repo_root_from_here() / "data" / "manifests" / dataset / "protocol_note.md"
    if not src.exists():
        return
    dst = out_dir / "protocol_note.md"
    if src.resolve() == dst.resolve() if dst.exists() else False:
        return
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def iter_files(root: Path, *, exts: Sequence[str]) -> Iterable[Path]:
    want = {e.lower() for e in exts}
    if not root.exists():
        return []
    return (p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in want)


def iter_images(root: Path, *, exts: Optional[set[str]] = None) -> Iterable[Path]:
    return iter_files(root, exts=sorted(exts or IMG_EXTS))


def infer_session(path: Path) -> int:
    for part in path.parts:
        m = re.search(r"(?:^|[_\-])(first|1st)(?:$|[_\-])", part, flags=re.IGNORECASE)
        if m:
            return 1
        m = re.search(r"(?:^|[_\-])(second|2nd)(?:$|[_\-])", part, flags=re.IGNORECASE)
        if m:
            return 2
        m = re.fullmatch(r"DS(\d+)", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"(?:session|sess|visit|capture)[_\- ]?(\d+)", part, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 1


def parse_last_int(text: str) -> Optional[int]:
    nums = re.findall(r"\d+", text)
    return int(nums[-1]) if nums else None


def split_by_subject(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float, *, subject_col: str = "subject_id") -> Dict[str, List[int]]:
    subjects = sorted(int(s) for s in df[subject_col].dropna().unique().tolist())
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

    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def assign_split(df: pd.DataFrame, split_map: Dict[str, List[int]], *, subject_col: str = "subject_id") -> pd.DataFrame:
    sid_to_split = {}
    for split_name, ids in split_map.items():
        for sid in ids:
            sid_to_split[int(sid)] = split_name
    out = df.copy()
    out["split"] = out[subject_col].map(lambda x: sid_to_split.get(int(x), None))
    return out


def choose_one_per_key(df: pd.DataFrame, group_cols: Sequence[str], *, sort_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    sort_columns = list(sort_cols or [])
    if not sort_columns:
        sort_columns = list(group_cols) + [c for c in ["session", "sample_id", "frame_id", "variant_rank", "path"] if c in out.columns]
    out = out.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return out.groupby(list(group_cols), as_index=False).first()


def write_split_json(out_dir: Path, split_map: Dict[str, List[int]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split.json").write_text(json.dumps(split_map, indent=2), encoding="utf-8")


def write_nested_pairs_bundle(
    out_dir: Path,
    split_map: Dict[str, List[int]],
    *,
    seed: int,
    neg_per_pos: int,
    manifest_path: Path,
    positive_pair_policy: str,
    negative_pair_policy: str,
    finger_col: str,
    resolved_data_dir: Path,
    same_finger_policy: object,
    max_pos_per_subject: Optional[int] = None,
    max_pos_per_finger: Optional[int] = None,
    pair_mode: Optional[str] = None,
) -> None:
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for split_name in DEFAULT_SPLITS:
        src = out_dir / f"pairs_{split_name}.csv"
        if src.exists():
            shutil.copy2(src, pairs_dir / src.name)

    split_meta = build_split_subjects_metadata(
        splits=split_map,
        seed=seed,
        neg_per_pos=neg_per_pos,
        impostors_per_pos=neg_per_pos,
        same_finger_policy=same_finger_policy,
        negative_pair_policy=negative_pair_policy,
        positive_pair_policy=positive_pair_policy,
        finger_col=finger_col,
        resolved_data_dir=resolved_data_dir,
        manifest_path=manifest_path,
        max_pos_per_subject=max_pos_per_subject,
        max_pos_per_finger=max_pos_per_finger,
        pair_mode=pair_mode,
    )
    validate_split_subjects_metadata(split_meta, context=f"{manifest_path.parent.name} split_subjects metadata")
    write_json(pairs_dir / "split_subjects.json", split_meta)


def write_pair_metadata(
    out_dir: Path,
    *,
    dataset: str,
    seed: int,
    neg_per_pos: int,
    finger_col: str,
    positive_pair_policy: str,
    negative_pair_policy: str,
    extra: dict,
) -> None:
    meta = build_pairs_split_build_meta(
        dataset=dataset,
        seed=int(seed),
        neg_per_pos=int(neg_per_pos),
        impostors_per_pos=int(neg_per_pos),
        finger_col=finger_col,
        positive_pair_policy=positive_pair_policy,
        negative_pair_policy=negative_pair_policy,
        extra=extra,
    )
    validate_pairs_split_build_meta(meta, context=f"{dataset} pairs_split_build metadata")
    write_json(out_dir / "pairs_split_build.meta.json", meta)


def _empty_pair_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_PAIR_COLUMNS)


def write_pairs_csvs(out_dir: Path, *, dataset: str, pos: pd.DataFrame, neg: pd.DataFrame) -> None:
    pos = canonicalize_pairs_df(pos) if not pos.empty else _empty_pair_df()
    if not pos.empty:
        pos = validate_canonical_pairs_df(pos, context=f"{dataset} positive pairs", require_exact_columns=True, require_non_empty=True)
    neg = canonicalize_pairs_df(neg) if not neg.empty else _empty_pair_df()
    if not neg.empty:
        neg = validate_canonical_pairs_df(neg, context=f"{dataset} negative pairs", require_exact_columns=True, require_non_empty=True)

    pos.to_csv(out_dir / "pairs_pos.csv", index=False)
    neg.to_csv(out_dir / "pairs_neg.csv", index=False)

    for split_name in DEFAULT_SPLITS:
        parts = []
        if not pos.empty:
            parts.append(pos[pos["split"] == split_name])
        if not neg.empty:
            parts.append(neg[neg["split"] == split_name])
        combined = pd.concat(parts, ignore_index=True) if parts else _empty_pair_df()
        combined = canonicalize_pairs_df(combined, split=split_name)
        combined = validate_canonical_pairs_df(
            combined,
            context=f"{dataset}/{split_name} canonical pairs",
            expected_split=split_name,
            require_exact_columns=True,
            require_non_empty=True,
        )
        combined.to_csv(out_dir / f"pairs_{split_name}.csv", index=False)


def build_basic_sanity(df: pd.DataFrame, split_map: Dict[str, List[int]], pos: pd.DataFrame, neg: pd.DataFrame, *, subject_col: str = "subject_id") -> dict:
    split_sets = {k: set(v) for k, v in split_map.items()}
    disjoint_ok = all(not (split_sets[a] & split_sets[b]) for a in DEFAULT_SPLITS for b in DEFAULT_SPLITS if a < b)

    leak_rows = 0
    for split_name in DEFAULT_SPLITS:
        ids = split_sets[split_name]
        leak_rows += int(((df["split"] == split_name) & (~df[subject_col].isin(ids))).sum())

    pos_bad = int((pos["subject_a"] != pos["subject_b"]).sum()) if not pos.empty else 0
    neg_bad = int((neg["subject_a"] == neg["subject_b"]).sum()) if not neg.empty else 0

    return {
        "disjoint_subject_splits": bool(disjoint_ok),
        "leak_rows": int(leak_rows),
        "positive_subject_mismatch": int(pos_bad),
        "negative_same_subject": int(neg_bad),
        "ok": bool(disjoint_ok and leak_rows == 0 and pos_bad == 0 and neg_bad == 0),
    }


def stats_with_pairs(df: pd.DataFrame, pos: pd.DataFrame, neg: pd.DataFrame, *, extra: Optional[dict] = None) -> dict:
    stats = {
        "manifest_rows": int(len(df)),
        "unique_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
        "capture_counts": {str(k): int(v) for k, v in df["capture"].value_counts().to_dict().items()} if "capture" in df.columns else {},
        "source_modality_counts": {str(k): int(v) for k, v in df["source_modality"].value_counts().to_dict().items()} if "source_modality" in df.columns else {},
        "pos_pairs": int(len(pos)),
        "neg_pairs": int(len(neg)),
        "pos_by_split": {str(k): int(v) for k, v in pos["split"].value_counts().to_dict().items()} if not pos.empty else {},
        "neg_by_split": {str(k): int(v) for k, v in neg["split"].value_counts().to_dict().items()} if not neg.empty else {},
    }
    if extra:
        stats.update(extra)
    return stats


def _pair_report(df: pd.DataFrame) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"rows": int(len(df)), "columns": list(df.columns)}
    rep["canonical_columns"] = bool(list(df.columns) == CANONICAL_PAIR_COLUMNS)
    rep["by_split"] = {str(k): int(v) for k, v in df["split"].value_counts().to_dict().items()} if "split" in df.columns else {}
    rep["label_values"] = sorted(int(v) for v in df["label"].dropna().unique().tolist()) if "label" in df.columns and not df.empty else []
    return rep


def _first_missing_paths(series: Iterable[str], limit: int = 3) -> List[str]:
    missing: List[str] = []
    for path_str in series:
        if len(missing) >= limit:
            break
        try:
            if not Path(str(path_str)).exists():
                missing.append(str(path_str))
        except Exception:
            missing.append(str(path_str))
    return missing


def build_active_compatible_sanity_report(
    *,
    dataset: str,
    data_dir: Path,
    raw_root: Path,
    manifest_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    extra_manifest: Optional[Mapping[str, Any]] = None,
    extra_checks: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "paths": {
            "data_dir": str(data_dir),
            "raw_root": str(raw_root),
            "manifest": str(data_dir / "manifest.csv"),
            "pairs_pos": str(data_dir / "pairs_pos.csv"),
            "pairs_neg": str(data_dir / "pairs_neg.csv"),
            "stats": str(data_dir / "stats.json"),
        },
        "manifest": {
            "rows": int(len(manifest_df)),
            "unique_subjects": int(manifest_df["subject_id"].nunique()) if "subject_id" in manifest_df.columns else 0,
            "frgp_values": sorted(int(v) for v in manifest_df["frgp"].dropna().unique().tolist()) if "frgp" in manifest_df.columns and not manifest_df.empty else [],
            "ppi_values": sorted(int(v) for v in manifest_df["ppi"].dropna().unique().tolist()) if "ppi" in manifest_df.columns and not manifest_df.empty else [],
            "capture_counts": {str(k): int(v) for k, v in manifest_df["capture"].value_counts().to_dict().items()} if "capture" in manifest_df.columns else {},
            "split_values": sorted(str(v) for v in manifest_df["split"].dropna().unique().tolist()) if "split" in manifest_df.columns else [],
            "source_modality_counts": {str(k): int(v) for k, v in manifest_df["source_modality"].value_counts().to_dict().items()} if "source_modality" in manifest_df.columns else {},
        },
        "pairs": {
            "pos": _pair_report(pos_df),
            "neg": _pair_report(neg_df),
        },
        "checks": {
            "remap_counts": {"manifest": {"path": 0}, "pos": {"path_a": 0, "path_b": 0}, "neg": {"path_a": 0, "path_b": 0}},
            "repaired_files_inplace": {},
        },
        "problems": [],
    }

    if extra_manifest:
        report["manifest"].update(dict(extra_manifest))
    if extra_checks:
        report["checks"].update(dict(extra_checks))

    required_cols = {"dataset", "capture", "subject_id", "frgp", "ppi", "path"}
    missing_manifest_cols = sorted(required_cols - set(manifest_df.columns))
    if missing_manifest_cols:
        report["problems"].append(f"manifest missing columns: {missing_manifest_cols}")

    if "split" in manifest_df.columns and "subject_id" in manifest_df.columns:
        subj_splits = manifest_df.groupby("subject_id")["split"].nunique()
        leaked = subj_splits[subj_splits > 1]
        report["checks"]["identity_leakage_subjects"] = int(len(leaked))
        if len(leaked) > 0:
            report["problems"].append(f"IDENTITY LEAKAGE: {len(leaked)} subjects appear in multiple splits.")
    else:
        report["checks"]["identity_leakage_subjects"] = 0

    if "path" in manifest_df.columns:
        missing_manifest = _first_missing_paths(manifest_df["path"].head(200).tolist())
        report["checks"]["missing_files_in_first_200"] = int(len(missing_manifest))
        if missing_manifest:
            report["problems"].append(f"Some files do not exist (first 200 sample). Example: {missing_manifest}")
    else:
        report["checks"]["missing_files_in_first_200"] = 0

    for kind, pairs_df, expected_label in (("pos", pos_df, 1), ("neg", neg_df, 0)):
        if list(pairs_df.columns) != CANONICAL_PAIR_COLUMNS:
            report["problems"].append(f"{kind}: non-canonical columns {list(pairs_df.columns)}")
        if "label" in pairs_df.columns and not pairs_df.empty:
            bad_label = pairs_df[pairs_df["label"] != expected_label]
            if not bad_label.empty:
                report["problems"].append(f"{kind}: {len(bad_label)} rows have wrong label (expected {expected_label}).")
        if {"subject_a", "subject_b"}.issubset(pairs_df.columns):
            if kind == "pos":
                bad = pairs_df[pairs_df["subject_a"] != pairs_df["subject_b"]]
                if not bad.empty:
                    report["problems"].append(f"{kind}: {len(bad)} pairs are not same-subject.")
            else:
                bad = pairs_df[pairs_df["subject_a"] == pairs_df["subject_b"]]
                if not bad.empty:
                    report["problems"].append(f"{kind}: {len(bad)} pairs are same-subject but should be negative.")
        if {"path_a", "path_b"}.issubset(pairs_df.columns):
            missing_pairs = []
            for path_a, path_b in pairs_df[["path_a", "path_b"]].head(200).itertuples(index=False):
                if not Path(str(path_a)).exists() or not Path(str(path_b)).exists():
                    missing_pairs.append((str(path_a), str(path_b)))
                    if len(missing_pairs) >= 1:
                        break
            if missing_pairs:
                report["problems"].append(f"{kind}: missing files in first 200 pairs. Example: {missing_pairs[:1]}")

    split_subjects_path = data_dir / "pairs" / "split_subjects.json"
    if split_subjects_path.exists():
        try:
            validate_split_subjects_metadata(json.loads(split_subjects_path.read_text(encoding="utf-8")), context=str(split_subjects_path))
        except Exception as exc:
            report["problems"].append(f"split_subjects metadata invalid: {exc}")

    pair_build_meta_path = data_dir / "pairs_split_build.meta.json"
    if pair_build_meta_path.exists():
        try:
            validate_pairs_split_build_meta(json.loads(pair_build_meta_path.read_text(encoding="utf-8")), context=str(pair_build_meta_path))
        except Exception as exc:
            report["problems"].append(f"pairs_split_build metadata invalid: {exc}")

    return report


def write_stats_and_sanity(
    out_dir: Path,
    *,
    dataset: str,
    raw_root: Path,
    manifest_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    stats: dict,
    extra_manifest: Optional[Mapping[str, Any]] = None,
    extra_checks: Optional[Mapping[str, Any]] = None,
) -> None:
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    sanity = build_active_compatible_sanity_report(
        dataset=dataset,
        data_dir=out_dir,
        raw_root=raw_root,
        manifest_df=manifest_df,
        pos_df=pos_df,
        neg_df=neg_df,
        extra_manifest=extra_manifest,
        extra_checks=extra_checks,
    )
    (out_dir / "sanity_report.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
