from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

import apps.api.scanner_capture as scanner_capture_module
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
    helper_path: Path | None = None,
) -> tuple[Path, Path]:
    incoming = tmp_path / "incoming"
    normalized = tmp_path / "normalized"
    twain_raw = tmp_path / "twain_raw"
    incoming.mkdir()

    monkeypatch.setenv("SCANNER_CAPTURE_DIR", str(incoming))
    monkeypatch.setenv("SCANNER_NORMALIZED_DIR", str(normalized))
    monkeypatch.setenv("SCANNER_TWAIN_RAW_DIR", str(twain_raw))
    monkeypatch.setenv("SCANNER_TWAIN_HELPER_PATH", str(helper_path or tmp_path / "missing_helper.exe"))
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


def _fake_helper_path(tmp_path: Path) -> Path:
    helper = tmp_path / "biometrika_twain_capture.exe"
    helper.write_text("fake helper", encoding="utf-8")
    return helper


def _helper_status_payload(*, ok: bool = True) -> dict[str, object]:
    return {
        "ok": ok,
        "mode": "status",
        "architecture": "x86",
        "twain_available": ok,
        "source_detected": ok,
        "source_name": "TWAIN Biometrika Driver" if ok else None,
        "twain_32_module": r"C:\WINDOWS\TWAIN_32.dll",
        "sources": [{"product_name": "TWAIN Biometrika Driver"}] if ok else [],
        "message": None if ok else "source not found",
    }


def _completed(args: list[str], payload: dict[str, object], *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout=json.dumps(payload), stderr="")


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


def test_scanner_status_includes_saved_file_bridge_when_incoming_dir_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scanner_dirs(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/api/scanner/status")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["saved_file_bridge_available"] is True
    assert "saved_file_bridge" in payload["available_modes"]
    assert payload["direct_capture_available"] is False


def test_scanner_status_reports_twain_available_when_helper_status_is_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        return _completed(args, _helper_status_payload(ok=True))

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.get("/api/scanner/status")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["direct_capture_available"] is True
    assert payload["twain_source_detected"] is True
    assert payload["device_name"] == "TWAIN Biometrika Driver"
    assert payload["active_mode"] == "twain"
    assert payload["enabled"] is True


def test_scanner_status_enabled_when_twain_available_without_saved_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    incoming, _normalized = _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    incoming.rmdir()
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        assert args == [str(helper), "--status"]
        assert timeout_seconds == 8.0
        return _completed(args, _helper_status_payload(ok=True))

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.get("/api/scanner/status")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["direct_capture_available"] is True
    assert payload["saved_file_bridge_available"] is False
    assert payload["enabled"] is True
    assert payload["direct_capture_enabled"] is True
    assert payload["saved_file_bridge_enabled"] is False
    assert payload["available_modes"] == ["twain"]


def test_scanner_status_handles_helper_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del args, timeout_seconds
        raise OSError("helper is not executable")

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.get("/api/scanner/status")

    assert response.status_code == 200, response.text
    payload = response.json()
    twain_status = payload["diagnostics"]["twain"]
    assert payload["direct_capture_available"] is False
    assert payload["last_error"] == "TWAIN helper could not be executed."
    assert twain_status["available"] is False
    assert twain_status["source_detected"] is False
    assert twain_status["diagnostics"]["stage"] == "status_subprocess_oserror"
    assert twain_status["diagnostics"]["error"] == "helper is not executable"


@pytest.mark.parametrize("helper_exists", [False, True])
def test_scanner_status_reports_twain_unavailable_when_helper_missing_or_fails(
    helper_exists: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path) if helper_exists else tmp_path / "missing_helper.exe"
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    if helper_exists:
        def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
            del timeout_seconds
            return _completed(args, _helper_status_payload(ok=False), returncode=1)

        monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)

    client = TestClient(app)

    response = client.get("/api/scanner/status")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["direct_capture_available"] is False
    assert "twain" not in payload["available_modes"]
    assert payload["last_error"]


def test_scanner_capture_auto_uses_twain_when_helper_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _incoming, normalized = _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)
    capture_args: list[str] = []

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))
        capture_args.extend(args)
        output_dir = Path(args[args.index("--output-dir") + 1])
        raw_path = output_dir / "capture.bmp"
        _write_image(raw_path, color=72)
        return _completed(
            args,
            {
                "ok": True,
                "mode": "capture",
                "provider": "twain",
                "source_name": "TWAIN Biometrika Driver",
                "show_ui": False,
                "transfer_mechanism": "native",
                "output_path": str(raw_path),
                "output_size_bytes": raw_path.stat().st_size,
                "duration_ms": 42,
                "image": {"format": "bmp", "width": 16, "height": 16, "bits_per_pixel": 8, "compression": "BI_RGB"},
                "events": [],
            },
        )

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.post("/api/scanner/capture", json={"mode": "auto", "fallback_allowed": False})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode_used"] == "twain"
    assert payload["direct_capture"] is True
    assert payload["device"]["name"] == "TWAIN Biometrika Driver"
    assert (normalized / Path(payload["normalized_file"]["path"]).name).read_bytes().startswith(PNG_SIGNATURE)
    asset_response = client.get(payload["normalized_url"])
    assert asset_response.status_code == 200
    assert asset_response.content.startswith(PNG_SIGNATURE)
    assert capture_args[capture_args.index("--settle-after-enable-ms") + 1] == "1500"


def test_scanner_capture_twain_does_not_fallback_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    incoming, _normalized = _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    _write_image(incoming / "saved.bmp")
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))
        return _completed(
            args,
            {"ok": False, "mode": "capture", "provider": "twain", "error_code": "transfer_failed", "message": "boom"},
            returncode=1,
        )

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.post("/api/scanner/capture", json={"mode": "twain", "fallback_allowed": False})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error_code"] == "helper_failed"
    assert payload["mode_requested"] == "twain"


def test_scanner_capture_auto_falls_back_only_when_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming, _normalized = _configure_scanner_dirs(monkeypatch, tmp_path)
    _write_image(incoming / "saved.bmp", color=180)
    client = TestClient(app)

    blocked = client.post("/api/scanner/capture", json={"mode": "auto", "fallback_allowed": False})
    allowed = client.post("/api/scanner/capture", json={"mode": "auto", "fallback_allowed": True})

    assert blocked.status_code == 200, blocked.text
    assert blocked.json()["ok"] is False
    assert blocked.json()["error_code"] == "twain_unavailable"
    assert allowed.status_code == 200, allowed.text
    allowed_payload = allowed.json()
    assert allowed_payload["ok"] is True
    assert allowed_payload["mode_used"] == "saved_file_bridge"
    assert allowed_payload["direct_capture"] is False
    assert allowed_payload["warning"] == "TWAIN direct capture failed/unavailable; saved-file bridge fallback was used."


def test_scanner_capture_helper_timeout_returns_capture_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))
        raise subprocess.TimeoutExpired(args, timeout_seconds)

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.post("/api/scanner/capture", json={"mode": "twain", "fallback_allowed": False})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error_code"] == "capture_timeout"


def test_scanner_capture_helper_oserror_returns_helper_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))
        raise OSError("CreateProcess failed")

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.post("/api/scanner/capture", json={"mode": "twain", "fallback_allowed": False})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error_code"] == "helper_failed"
    assert payload["message"] == "TWAIN helper could not be executed."
    assert payload["diagnostics"]["stage"] == "capture_subprocess_oserror"
    assert payload["diagnostics"]["error"] == "CreateProcess failed"


def test_scanner_capture_output_dir_creation_failure_returns_helper_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))
        pytest.fail("capture helper should not run when output directory creation fails")

    def fake_mkdtemp(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise OSError("cannot create temp directory")

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    monkeypatch.setattr(scanner_capture_module.tempfile, "mkdtemp", fake_mkdtemp)
    client = TestClient(app)

    response = client.post("/api/scanner/capture", json={"mode": "twain", "fallback_allowed": False})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error_code"] == "helper_failed"
    assert payload["message"] == "Could not create TWAIN helper output directory."
    assert payload["diagnostics"]["stage"] == "create_twain_output_dir"
    assert payload["diagnostics"]["error"] == "cannot create temp directory"
    assert payload["diagnostics"]["twain_raw_dir"] == str(tmp_path / "twain_raw")


def test_scanner_capture_passes_show_ui_and_timeout_to_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)
    capture_args: list[str] = []
    capture_timeout_seconds: list[float] = []

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))

        capture_args.extend(args)
        capture_timeout_seconds.append(timeout_seconds)
        output_dir = Path(args[args.index("--output-dir") + 1])
        raw_path = output_dir / "capture.bmp"
        _write_image(raw_path, color=72)
        return _completed(
            args,
            {
                "ok": True,
                "mode": "capture",
                "provider": "twain",
                "source_name": "TWAIN Biometrika Driver",
                "output_path": str(raw_path),
                "duration_ms": 42,
            },
        )

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.post(
        "/api/scanner/capture",
        json={"mode": "twain", "fallback_allowed": False, "show_ui": True, "timeout_ms": 1234},
    )

    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True
    assert capture_args[capture_args.index("--show-ui") + 1] == "true"
    assert capture_args[capture_args.index("--timeout-ms") + 1] == "1234"
    assert capture_args[capture_args.index("--settle-after-enable-ms") + 1] == "0"
    assert capture_timeout_seconds == pytest.approx([6.234])


def test_scanner_capture_passes_requested_headless_settle_to_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)
    capture_args: list[str] = []

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))

        capture_args.extend(args)
        output_dir = Path(args[args.index("--output-dir") + 1])
        raw_path = output_dir / "capture.bmp"
        _write_image(raw_path, color=72)
        return _completed(
            args,
            {
                "ok": True,
                "mode": "capture",
                "provider": "twain",
                "source_name": "TWAIN Biometrika Driver",
                "output_path": str(raw_path),
                "duration_ms": 42,
                "settle_after_enable_ms": 2000,
                "xferready_waited_for_settle": True,
            },
        )

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.post(
        "/api/scanner/capture",
        json={"mode": "twain", "fallback_allowed": False, "show_ui": False, "settle_after_enable_ms": 2000},
    )

    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True
    assert capture_args[capture_args.index("--settle-after-enable-ms") + 1] == "2000"


@pytest.mark.parametrize(
    ("requested_settle_after_enable_ms", "expected_settle_after_enable_ms"),
    [(-50, "0"), (50000, "10000")],
)
def test_scanner_capture_clamps_invalid_settle_after_enable_ms(
    requested_settle_after_enable_ms: int,
    expected_settle_after_enable_ms: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)
    capture_args: list[str] = []

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))

        capture_args.extend(args)
        output_dir = Path(args[args.index("--output-dir") + 1])
        raw_path = output_dir / "capture.bmp"
        _write_image(raw_path, color=72)
        return _completed(
            args,
            {
                "ok": True,
                "mode": "capture",
                "provider": "twain",
                "source_name": "TWAIN Biometrika Driver",
                "output_path": str(raw_path),
                "duration_ms": 42,
            },
        )

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.post(
        "/api/scanner/capture",
        json={
            "mode": "twain",
            "fallback_allowed": False,
            "show_ui": False,
            "settle_after_enable_ms": requested_settle_after_enable_ms,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True
    assert capture_args[capture_args.index("--settle-after-enable-ms") + 1] == expected_settle_after_enable_ms


def test_scanner_capture_rejects_invalid_helper_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))
        return subprocess.CompletedProcess(args, 0, stdout="not json", stderr="")

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.post("/api/scanner/capture", json={"mode": "twain", "fallback_allowed": False})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error_code"] == "invalid_helper_output"


def test_scanner_capture_rejects_helper_output_outside_safe_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    outside = tmp_path / "outside.bmp"
    _write_image(outside)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))
        return _completed(args, {"ok": True, "output_path": str(outside), "duration_ms": 5})

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.post("/api/scanner/capture", json={"mode": "twain", "fallback_allowed": False})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error_code"] == "invalid_helper_output"


@pytest.mark.parametrize("case", ["missing", "non_image"])
def test_scanner_capture_rejects_helper_output_missing_or_non_image(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _fake_helper_path(tmp_path)
    _configure_scanner_dirs(monkeypatch, tmp_path, helper_path=helper)
    monkeypatch.setattr(scanner_capture_module, "_is_windows", lambda: True)

    def fake_run(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        if "--status" in args:
            return _completed(args, _helper_status_payload(ok=True))
        output_dir = Path(args[args.index("--output-dir") + 1])
        raw_path = output_dir / "capture.bmp"
        if case == "non_image":
            raw_path.write_text("not an image", encoding="utf-8")
        return _completed(args, {"ok": True, "output_path": str(raw_path), "duration_ms": 5})

    monkeypatch.setattr(scanner_capture_module, "_run_twain_helper", fake_run)
    client = TestClient(app)

    response = client.post("/api/scanner/capture", json={"mode": "twain", "fallback_allowed": False})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error_code"] == "invalid_helper_output"


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
