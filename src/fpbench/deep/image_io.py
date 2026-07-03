from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:/")


def _strip_file_uri(raw_path: str | Path) -> str:
    value = str(raw_path).strip().strip('"')
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    return value


def _norm(raw_path: str | Path) -> str:
    return _strip_file_uri(raw_path).replace("\\", "/")


def _existing(candidates: Iterable[Path]) -> Path | None:
    seen: set[str] = set()
    for candidate in candidates:
        try:
            candidate = candidate.expanduser()
        except Exception:
            pass
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return None


def _suffixes_from_known_roots(normalized: str) -> list[str]:
    """Return relative suffixes useful for remapping Windows paths on Linux/Kaggle.

    Examples:
        C:/fingerprint-research/data/raw/NIST/... ->
        [data/raw/NIST/..., raw/NIST/..., NIST/...]
    """
    s = normalized.lstrip("/")
    lowered = s.lower()
    suffixes: list[str] = []
    for marker in ("data/raw/", "data/processed/", "data/manifests/", "data/"):
        idx = lowered.find(marker)
        if idx >= 0:
            suffix = s[idx:]
            if suffix not in suffixes:
                suffixes.append(suffix)
            after_data = suffix[len("data/") :] if suffix.startswith("data/") else suffix
            if after_data and after_data not in suffixes:
                suffixes.append(after_data)
    for marker in ("raw/", "processed/", "manifests/"):
        idx = lowered.find(marker)
        if idx >= 0:
            suffix = s[idx:]
            if suffix not in suffixes:
                suffixes.append(suffix)
            after_marker = suffix[len(marker) :]
            if after_marker and after_marker not in suffixes:
                suffixes.append(after_marker)
    return suffixes


def resolve_fingerprint_path(
    raw_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    data_root: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a fingerprint image path across local Windows and Kaggle/Linux layouts.

    Pair CSVs in this project can contain absolute Windows paths such as
    ``C:\\fingerprint-research\\data\\raw\\NIST\\...``.  On Linux/Kaggle those
    paths are invalid, but the relative suffix under ``data/`` or ``raw/`` is
    still meaningful. This resolver tries direct paths first, then remaps the
    suffix against ``repo_root`` and/or ``data_root``.

    Args:
        raw_path: Path from a manifest or pairs CSV. ``file:/C:/...`` URIs are
            accepted.
        repo_root: Repository root. Used for ``repo_root/data/...`` remapping.
        data_root: Optional data root. It may point either to repo root, the
            ``data`` directory, or a raw-data directory.
        must_exist: If true, raise ``FileNotFoundError`` when no candidate exists.

    Returns:
        A resolved ``Path``. When ``must_exist=False`` and no candidate exists,
        returns the most useful remapped candidate if possible, otherwise the
        normalized direct path.
    """
    normalized = _norm(raw_path)
    direct_candidates: list[Path] = []

    try:
        direct_candidates.append(Path(_strip_file_uri(raw_path)))
    except Exception:
        pass
    direct_candidates.append(Path(normalized))

    found = _existing(direct_candidates)
    if found is not None:
        return found

    suffixes = _suffixes_from_known_roots(normalized)
    remapped: list[Path] = []
    roots: list[Path] = []
    if repo_root is not None:
        roots.append(Path(repo_root))
        roots.append(Path(repo_root) / "data")
    if data_root is not None:
        roots.append(Path(data_root))
        roots.append(Path(data_root) / "data")

    for root in roots:
        for suffix in suffixes:
            suffix_path = Path(suffix)
            remapped.append(root / suffix_path)
            if suffix.startswith("data/"):
                remapped.append(root / Path(suffix[len("data/") :]))

    found = _existing(remapped)
    if found is not None:
        return found

    if not must_exist:
        if remapped:
            return remapped[0]
        if normalized and _WINDOWS_DRIVE_RE.match(normalized):
            return Path(normalized)
        return Path(_strip_file_uri(raw_path))

    checked = [str(p) for p in [*direct_candidates, *remapped]][:30]
    raise FileNotFoundError(
        f"Could not resolve fingerprint image path {raw_path!r}. "
        f"Checked first {len(checked)} candidates: {checked}"
    )


def load_grayscale_image(
    raw_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    data_root: str | Path | None = None,
) -> np.ndarray:
    """Load a fingerprint image as a uint8 grayscale numpy array."""
    path = resolve_fingerprint_path(raw_path, repo_root=repo_root, data_root=data_root, must_exist=True)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        return np.asarray(image, dtype=np.uint8)
