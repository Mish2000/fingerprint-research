"""PolyU Cross pair loading and relative-image-path resolution.

This module is intentionally small and dependency-light so it can be unit
tested in isolation and reused by the thin PolyU Cross zero-shot benchmark
runner (``pipelines/benchmark/run_polyu_cross_zero_shot.py``).

Responsibilities:

* Load the *existing* canonical PolyU Cross pair CSVs produced in Phase 2A.
  The runner never regenerates pairs; it only reads what is already on disk.
* Resolve the ``configured PolyU dataset root`` used to turn the relative
  ``path_a``/``path_b`` image references into absolute paths for scoring.
* Preserve compatibility with older pair schemas that stored absolute image
  paths (e.g. ``nist_sd300b``/``polyu_3d``) and lacked the newer PolyU Cross
  columns (``sample_uid_a/b``, ``modality_a/b``, ``session_a/b``,
  ``finger_unit_a/b``).

Nothing in this module mutates the manifest, the split files, or the pair
CSVs; it is read-only.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Avoid importing broken optional pandas accelerators in mixed NumPy
# environments (mirrors the guard used across pipelines/*).
import sys

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

DATASET = "polyu_cross"

# Environment override for the PolyU Cross dataset root. When set it wins over
# any value recorded in the pair-bundle metadata, which is useful when the
# dataset drive is mounted at a different location than when the manifest was
# built.
POLYU_CROSS_ROOT_ENV = "FPRJ_POLYU_CROSS_ROOT"

# Minimal columns every pair CSV must expose for benchmark scoring. Newer
# PolyU Cross columns are optional so older bundles keep loading.
REQUIRED_PAIR_COLUMNS = (
    "pair_id",
    "label",
    "split",
    "subject_a",
    "subject_b",
    "path_a",
    "path_b",
)

# Newer PolyU Cross identity/provenance columns. Absent in legacy bundles.
OPTIONAL_PAIR_COLUMNS = (
    "finger_unit_a",
    "finger_unit_b",
    "frgp",
    "modality_a",
    "modality_b",
    "session_a",
    "session_b",
    "sample_uid_a",
    "sample_uid_b",
)

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class PolyUCrossPairError(RuntimeError):
    """Raised when PolyU Cross pairs cannot be loaded or resolved safely."""


@dataclass(frozen=True)
class ResolvedRoot:
    """The resolved PolyU dataset root plus where it came from."""

    root: Optional[Path]
    source: str
    exists: bool


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_polyu_cross_root(
    manifest_dir: Path,
    *,
    override: Optional[str] = None,
) -> ResolvedRoot:
    """Resolve the dataset root used to expand relative image paths.

    Resolution order (first hit wins):

    1. Explicit ``override`` (e.g. a CLI ``--polyu_root`` argument).
    2. The ``FPRJ_POLYU_CROSS_ROOT`` environment variable.
    3. ``path_base`` recorded in ``pairs_split_build.meta.json``.
    4. ``resolved_data_dir`` recorded in ``pairs/split_subjects.json``.

    The root is returned even when it does not exist on the current machine so
    the caller can decide how to handle missing images (fail in strict mode,
    or record per-pair failures otherwise). ``root`` is ``None`` only when no
    source provided a value.
    """

    manifest_dir = Path(manifest_dir)

    candidates: list[tuple[str, Optional[str]]] = [
        ("override", override),
        (f"env:{POLYU_CROSS_ROOT_ENV}", os.environ.get(POLYU_CROSS_ROOT_ENV)),
    ]

    meta = _read_json(manifest_dir / "pairs_split_build.meta.json") or {}
    candidates.append(("pairs_split_build.meta.json:path_base", meta.get("path_base")))

    split_meta = _read_json(manifest_dir / "pairs" / "split_subjects.json") or {}
    candidates.append(
        ("pairs/split_subjects.json:resolved_data_dir", split_meta.get("resolved_data_dir"))
    )

    for source, value in candidates:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        root = Path(text).expanduser()
        return ResolvedRoot(root=root, source=source, exists=root.exists())

    return ResolvedRoot(root=None, source="unresolved", exists=False)


def looks_absolute(raw_path: str) -> bool:
    """True for absolute POSIX paths, Windows drive paths, or ``file:`` URIs."""

    text = str(raw_path)
    if text.startswith("file:"):
        return True
    if _WINDOWS_DRIVE_RE.match(text):
        return True
    # A leading root separator is an absolute POSIX path (or a UNC/rooted
    # Windows path). ``Path.is_absolute`` returns False for these on Windows
    # because no drive is present, so recognize them explicitly to keep legacy
    # bundles that stored rooted paths from being re-joined onto the dataset
    # root.
    if text[:1] in ("/", "\\"):
        return True
    return Path(text).is_absolute()


def _strip_file_uri(raw_path: str) -> str:
    text = str(raw_path)
    if text.startswith("file:"):
        text = text[len("file:") :]
        # "file:/C:/..." -> "/C:/..." -> "C:/..."
        if len(text) >= 3 and text[0] == "/" and text[2] == ":":
            text = text[1:]
    return text


def resolve_pair_image_path(raw_path: str, root: Optional[Path]) -> Path:
    """Resolve a single pair image reference to a concrete path.

    Absolute references (older bundles) are returned as-is. Relative
    references (PolyU Cross) are joined onto ``root``. Raising here only
    happens when a relative path is encountered without any configured root.
    """

    text = _strip_file_uri(str(raw_path))
    if looks_absolute(raw_path):
        return Path(text)
    if root is None:
        raise PolyUCrossPairError(
            "Cannot resolve relative PolyU image path "
            f"{raw_path!r} because no PolyU dataset root is configured. "
            f"Set --polyu_root or the {POLYU_CROSS_ROOT_ENV} environment variable."
        )
    return Path(root) / text


def load_polyu_cross_pairs(pairs_csv: Path) -> pd.DataFrame:
    """Load a canonical pair CSV, validating the minimal required columns.

    The returned frame keeps *all* original columns (so newer PolyU Cross
    identity columns survive) and adds a stable ``_row_order`` index used to
    re-align externally produced score rows without mutating pair identity.
    """

    pairs_csv = Path(pairs_csv)
    if not pairs_csv.exists():
        raise FileNotFoundError(f"PolyU Cross pairs CSV not found: {pairs_csv}")

    df = pd.read_csv(pairs_csv)
    missing = [column for column in REQUIRED_PAIR_COLUMNS if column not in df.columns]
    if missing:
        raise PolyUCrossPairError(
            f"Pairs CSV {pairs_csv} is missing required columns {missing}; found {list(df.columns)}"
        )

    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    df["_row_order"] = range(len(df))
    return df


def finger_position_series(df: pd.DataFrame) -> pd.Series:
    """Best-effort finger-position label for downstream schemas.

    PolyU Cross exposes the client/finger identifier as ``finger_unit_a`` and
    keeps ``frgp`` constant (0). Fall back through the columns other datasets
    use so this works for legacy bundles too.
    """

    for column in ("finger_unit_a", "frgp", "finger_id", "finger_position"):
        if column in df.columns:
            return df[column].astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def resolve_pairs_frame(df: pd.DataFrame, root: Optional[Path]) -> pd.DataFrame:
    """Add resolved absolute image paths and existence flags.

    Adds ``resolved_path_a``/``resolved_path_b`` (absolute strings) and
    ``path_a_exists``/``path_b_exists`` booleans. The original ``path_a`` and
    ``path_b`` columns are left untouched to preserve provenance.
    """

    out = df.copy()
    resolved_a: list[str] = []
    resolved_b: list[str] = []
    exists_a: list[bool] = []
    exists_b: list[bool] = []
    for raw_a, raw_b in zip(out["path_a"].astype(str), out["path_b"].astype(str)):
        pa = resolve_pair_image_path(raw_a, root)
        pb = resolve_pair_image_path(raw_b, root)
        resolved_a.append(str(pa))
        resolved_b.append(str(pb))
        exists_a.append(pa.exists())
        exists_b.append(pb.exists())
    out["resolved_path_a"] = resolved_a
    out["resolved_path_b"] = resolved_b
    out["path_a_exists"] = exists_a
    out["path_b_exists"] = exists_b
    return out


def balanced_limit(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Deterministically keep the first ``limit`` rows, balanced by label.

    Used only for smoke runs. Row order within each label is preserved, and
    the result is re-sorted by the original ``_row_order`` so downstream
    positional joins remain valid.
    """

    limit = int(limit)
    if limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)

    labels = sorted(int(x) for x in df["label"].dropna().unique().tolist())
    if len(labels) < 2:
        return df.head(limit).reset_index(drop=True)

    per_label = limit // len(labels)
    remainder = limit % len(labels)
    parts: list[pd.DataFrame] = []
    for index, label in enumerate(labels):
        want = per_label + (1 if index < remainder else 0)
        parts.append(df[df["label"] == label].head(want))
    out = pd.concat(parts, axis=0)
    if len(out) < limit:
        extra = df.drop(index=out.index, errors="ignore").head(limit - len(out))
        out = pd.concat([out, extra], axis=0)
    sort_key = "_row_order" if "_row_order" in out.columns else None
    if sort_key is not None:
        out = out.sort_values(sort_key, kind="mergesort")
    return out.reset_index(drop=True)


def iter_pair_split_csvs(manifest_dir: Path, splits: Iterable[str]) -> Iterable[tuple[str, Path]]:
    """Yield ``(split, pairs_csv_path)`` using the canonical resolution order.

    Prefers the flat ``pairs_<split>.csv`` and falls back to the nested
    compatibility copy under ``pairs/``.
    """

    manifest_dir = Path(manifest_dir)
    for split in splits:
        candidates = [
            manifest_dir / f"pairs_{split}.csv",
            manifest_dir / "pairs" / f"pairs_{split}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                yield split, candidate
                break
        else:
            yield split, candidates[0]
