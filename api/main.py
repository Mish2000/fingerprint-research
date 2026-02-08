from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException

# Ensure project root is importable (so `import src.*` works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.io import save_upload_to_temp
from api.schemas import MatchMethod, MatchResponse
from api.service import MatchService

app = FastAPI(title="Fingerprint Match API", version="0.1")

_service: Optional[MatchService] = None
_service_init_error: Optional[str] = None


@app.on_event("startup")
def _startup():
    global _service, _service_init_error
    try:
        _service = MatchService()
        _service_init_error = None
    except Exception as e:
        # Keep the server up, but error clearly on /match
        _service = None
        _service_init_error = repr(e)


@app.get("/health")
def health():
    return {"ok": _service is not None, "error": _service_init_error}


@app.post("/match", response_model=MatchResponse)
async def match(
    method: MatchMethod = Form(...),
    img_a: UploadFile = File(...),
    img_b: UploadFile = File(...),
    return_overlay: bool = Form(True),
    threshold: float | None = Form(None),
    capture_a: str | None = Form(None),
    capture_b: str | None = Form(None),
):
    if _service is None:
        raise HTTPException(status_code=500, detail=f"Service init failed: {_service_init_error}")

    path_a = await save_upload_to_temp(img_a, prefix="a", capture=capture_a)
    path_b = await save_upload_to_temp(img_b, prefix="b", capture=capture_b)

    try:
        return _service.match(
            method=method,
            path_a=str(path_a),
            path_b=str(path_b),
            threshold=threshold,
            return_overlay=bool(return_overlay),
            capture_a=capture_a,
            capture_b=capture_b,
            filename_a=img_a.filename,
            filename_b=img_b.filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))
    finally:
        # cleanup temp files
        try:
            path_a.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            path_b.unlink(missing_ok=True)
        except Exception:
            pass

