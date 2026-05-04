from __future__ import annotations

import os
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.scanner_capture import ScannerCapturePathError, resolve_normalized_capture_path

try:
    from PIL import Image
except Exception:  # pragma: no cover - depends on optional runtime package
    Image = None  # type: ignore[assignment]


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _configure_scanner_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    max_age_seconds: int = 3600,
) -> tuple[Path, Path]:
    incoming = tmp_path / "incoming"
    normalized = tmp_path / "normalized"
    incoming.mkdir()

    monkeypatch.setenv("SCANNER_CAPTURE_DIR", str(incoming))
    monkeypatch.setenv("SCANNER_NORMALIZED_DIR", str(normalized))
    monkeypatch.setenv("SCANNER_CAPTURE_GLOB", "*.tif;*.tiff;*.png;*.bmp;*.jpg;*.jpeg")
    monkeypatch.setenv("SCANNER_MAX_AGE_SECONDS", str(max_age_seconds))
    return incoming, normalized


def _write_image(path: Path, *, color: int = 128) -> None:
    if Image is None:
        pytest.skip("Pillow is required for scanner capture normalization tests.")

    image = Image.new("L", (16, 16), color=color)
    image.save(path)


def _set_mtime(path: Path, timestamp: float) -> None:
    os.utime(path, (timestamp, timestamp))


def test_scanner_import_returns_no_capture_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_scanner_dirs(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.post("/api/scanner/import-latest")

    assert response.status_code == 404
    assert response.json()["detail"] == "No saved scanner capture found in the configured folder."


def test_scanner_import_selects_newest_matching_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    incoming, normalized = _configure_scanner_dirs(monkeypatch, tmp_path)
    now = time.time()
    older = incoming / "older.png"
    newer = incoming / "newer.bmp"
    ignored_report = incoming / "report.txt"
    _write_image(older, color=32)
    _write_image(newer, color=96)
    ignored_report.write_text("diagnostic report", encoding="utf-8")
    _set_mtime(older, now - 120)
    _set_mtime(newer, now - 10)
    _set_mtime(ignored_report, now + 120)
    client = TestClient(app)

    response = client.post("/api/scanner/import-latest")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["original_filename"] == "newer.bmp"
    assert payload["mime_type"] == "image/png"
    assert payload["normalized_filename"].endswith(".png")
    assert (normalized / payload["normalized_filename"]).read_bytes().startswith(PNG_SIGNATURE)


def test_scanner_import_ignores_non_matching_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    incoming, _normalized = _configure_scanner_dirs(monkeypatch, tmp_path)
    now = time.time()
    matching = incoming / "matching.png"
    non_matching = incoming / "newer_report.txt"
    _write_image(matching, color=160)
    non_matching.write_text("newer report", encoding="utf-8")
    _set_mtime(matching, now - 60)
    _set_mtime(non_matching, now + 60)
    client = TestClient(app)

    response = client.post("/api/scanner/import-latest")

    assert response.status_code == 200, response.text
    assert response.json()["original_filename"] == "matching.png"


@pytest.mark.parametrize("extension", ["tif", "bmp"])
def test_scanner_import_normalizes_tiff_and_bmp_to_png(
    extension: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming, _normalized = _configure_scanner_dirs(monkeypatch, tmp_path)
    source = incoming / f"capture.{extension}"
    _write_image(source, color=220)
    client = TestClient(app)

    response = client.post("/api/scanner/import-latest")

    assert response.status_code == 200, response.text
    payload = response.json()
    asset_response = client.get(payload["normalized_url"])
    assert asset_response.status_code == 200
    assert asset_response.headers["content-type"].startswith("image/png")
    assert asset_response.content.startswith(PNG_SIGNATURE)


def test_scanner_import_rejects_stale_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    incoming, _normalized = _configure_scanner_dirs(monkeypatch, tmp_path, max_age_seconds=1)
    source = incoming / "old_capture.tif"
    _write_image(source)
    _set_mtime(source, time.time() - 60)
    client = TestClient(app)

    response = client.post("/api/scanner/import-latest")

    assert response.status_code == 409
    assert response.json()["detail"] == "Latest saved UMPI capture is too old. Save a new fingerprint scan and try again."


def test_scanner_capture_serving_blocks_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _incoming, normalized = _configure_scanner_dirs(monkeypatch, tmp_path)
    normalized.mkdir()
    (tmp_path / "secret.png").write_bytes(PNG_SIGNATURE + b"secret")
    client = TestClient(app)

    response = client.get("/api/scanner/captures/..%2Fsecret")

    assert response.status_code != 200
    assert response.content != PNG_SIGNATURE + b"secret"
    with pytest.raises(ScannerCapturePathError):
        resolve_normalized_capture_path("../secret")
