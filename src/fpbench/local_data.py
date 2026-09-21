"""Optional, read-only remapping of historical raw-data paths.

This maps paths, not people, splits, pairs, or processing parameters. It does not
ingest data or write caches into a dataset directory.
"""
import os
from pathlib import Path

RAW_ALIASES = {"sd300b": "NIST/sd300b", "sd300c": "NIST/sd300c",
               "PolyU_Hong_Kong": "PolyU Hong Kong"}


def configured_raw_path(raw_path: str | Path) -> Path | None:
    root_value = os.getenv("FPBENCH_DATASETS_ROOT")
    if not root_value:
        return None
    normalized = str(raw_path).replace("\\", "/")
    marker = "data/raw/"
    if marker not in normalized:
        return None
    relative = normalized.split(marker, 1)[1]
    first, _, tail = relative.partition("/")
    root = Path(root_value).resolve()
    target = (root / RAW_ALIASES.get(first, first) / tail).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Configured dataset path escapes its data root")
    return target
