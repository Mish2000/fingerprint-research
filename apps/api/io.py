from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import UploadFile


def _safe_capture(c: Optional[str]) -> Optional[str]:
    if c is None:
        return None
    c = c.strip().lower()
    if c in ("plain", "roll"):
        return c
    return None


async def save_upload_to_temp(upload: UploadFile, *, prefix: str, capture: Optional[str]) -> Path:
    """
    Save UploadFile to a temp file and return the path.
    We include 'plain'/'roll' in the filename when available so BaselineDL can infer capture type.
    """
    cap = _safe_capture(capture)
    suffix = Path(upload.filename or "").suffix
    if not suffix:
        suffix = ".png"

    name_prefix = f"fp_{prefix}_"
    if cap:
        name_prefix += f"{cap}_"

    tmp = tempfile.NamedTemporaryFile(delete=False, prefix=name_prefix, suffix=suffix)
    try:
        data = await upload.read()
        tmp.write(data)
        tmp.flush()
    finally:
        tmp.close()

    return Path(tmp.name)
