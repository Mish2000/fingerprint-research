from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

try:  # Pillow is used only for the local scanner-file bridge.
    from PIL import Image, ImageOps, UnidentifiedImageError
except Exception:  # pragma: no cover - exercised only in environments without Pillow
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

    class UnidentifiedImageError(OSError):
        pass


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CAPTURE_DIR = "data/scanner_captures/incoming"
DEFAULT_NORMALIZED_DIR = "data/scanner_captures/normalized"
DEFAULT_TWAIN_RAW_DIR = "data/scanner_captures/twain_raw"
DEFAULT_TWAIN_HELPER_PATH = "tools/biometrika_capture/bin/x86/biometrika_twain_capture.exe"
DEFAULT_CAPTURE_GLOB = "*.tif;*.tiff;*.png;*.bmp;*.jpg;*.jpeg"
DEFAULT_MAX_AGE_SECONDS = 3600
BIOMETRIKA_TWAIN_SOURCE_NAME = "TWAIN Biometrika Driver"

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}
VALID_HELPER_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CAPTURE_ID_PATTERN = re.compile(r"^scanner_\d{8}_\d{6}_[a-f0-9]{8}(?:\.png)?$")

NO_CAPTURE_MESSAGE = "No saved scanner capture found in the configured folder."
STALE_CAPTURE_MESSAGE = "Latest saved UMPI capture is too old. Save a new fingerprint scan and try again."
DEFAULT_HEADLESS_SETTLE_AFTER_ENABLE_MS = 1500
MIN_SETTLE_AFTER_ENABLE_MS = 0
MAX_SETTLE_AFTER_ENABLE_MS = 10000


class ScannerCaptureError(Exception):
    """Base error for the scanner capture bridge."""


class NoScannerCaptureFoundError(ScannerCaptureError):
    pass


class StaleScannerCaptureError(ScannerCaptureError):
    pass


class ScannerCaptureNormalizationError(ScannerCaptureError):
    pass


class ScannerCapturePathError(ScannerCaptureError):
    pass


class ScannerNormalizedCaptureNotFoundError(ScannerCaptureError):
    pass


class ScannerCaptureProviderError(ScannerCaptureError):
    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class ScannerCaptureConfig:
    capture_dir: Path
    normalized_dir: Path
    twain_raw_dir: Path
    twain_helper_path: Path
    capture_glob: str
    max_age_seconds: int
    capture_dir_display: str
    normalized_dir_display: str
    twain_raw_dir_display: str
    twain_helper_path_display: str


@dataclass(frozen=True)
class CaptureCandidate:
    path: Path
    stat: os.stat_result


def _configured_path(env_name: str, default_value: str) -> tuple[Path, str]:
    raw_value = os.getenv(env_name, default_value).strip() or default_value
    path = Path(raw_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path, raw_value


def _display_path(path: Path, raw_value: str) -> str:
    raw_path = Path(raw_value)
    if raw_value and not raw_path.is_absolute():
        return raw_value.replace("\\", "/")

    try:
        return path.resolve(strict=False).relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return f".../{path.name}" if path.name else "configured scanner capture folder"


def _configured_max_age() -> int:
    raw_value = os.getenv("SCANNER_MAX_AGE_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_MAX_AGE_SECONDS

    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_AGE_SECONDS

    return value if value >= 0 else DEFAULT_MAX_AGE_SECONDS


def load_scanner_capture_config() -> ScannerCaptureConfig:
    capture_dir, capture_dir_raw = _configured_path("SCANNER_CAPTURE_DIR", DEFAULT_CAPTURE_DIR)
    normalized_dir, normalized_dir_raw = _configured_path("SCANNER_NORMALIZED_DIR", DEFAULT_NORMALIZED_DIR)
    twain_raw_dir, twain_raw_dir_raw = _configured_path("SCANNER_TWAIN_RAW_DIR", DEFAULT_TWAIN_RAW_DIR)
    twain_helper_path, twain_helper_path_raw = _configured_path(
        "SCANNER_TWAIN_HELPER_PATH",
        DEFAULT_TWAIN_HELPER_PATH,
    )
    capture_glob = os.getenv("SCANNER_CAPTURE_GLOB", DEFAULT_CAPTURE_GLOB).strip() or DEFAULT_CAPTURE_GLOB

    return ScannerCaptureConfig(
        capture_dir=capture_dir,
        normalized_dir=normalized_dir,
        twain_raw_dir=twain_raw_dir,
        twain_helper_path=twain_helper_path,
        capture_glob=capture_glob,
        max_age_seconds=_configured_max_age(),
        capture_dir_display=_display_path(capture_dir, capture_dir_raw),
        normalized_dir_display=_display_path(normalized_dir, normalized_dir_raw),
        twain_raw_dir_display=_display_path(twain_raw_dir, twain_raw_dir_raw),
        twain_helper_path_display=_display_path(twain_helper_path, twain_helper_path_raw),
    )


def _glob_patterns(capture_glob: str) -> list[str]:
    patterns = [item.strip() for item in capture_glob.split(";") if item.strip()]
    return patterns or [item.strip() for item in DEFAULT_CAPTURE_GLOB.split(";")]


def _matches_capture_glob(path: Path, patterns: list[str]) -> bool:
    name = path.name.lower()
    return any(fnmatch.fnmatchcase(name, pattern.lower()) for pattern in patterns)


def find_latest_capture(config: ScannerCaptureConfig | None = None) -> CaptureCandidate | None:
    resolved_config = config or load_scanner_capture_config()
    capture_dir = resolved_config.capture_dir
    if not capture_dir.is_dir():
        return None

    patterns = _glob_patterns(resolved_config.capture_glob)
    candidates: list[CaptureCandidate] = []
    for child in capture_dir.iterdir():
        if not child.is_file():
            continue
        if child.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if not _matches_capture_glob(child, patterns):
            continue
        try:
            candidates.append(CaptureCandidate(path=child, stat=child.stat()))
        except OSError:
            continue

    if not candidates:
        return None

    return max(candidates, key=lambda item: (item.stat.st_mtime_ns, item.path.name.lower()))


def _iso_modified_at(stat: os.stat_result) -> str:
    return datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")


def _age_seconds(stat: os.stat_result, now: float | None = None) -> float:
    return round(max(0.0, (now if now is not None else time.time()) - stat.st_mtime), 3)


def _capture_metadata(
    candidate: CaptureCandidate,
    config: ScannerCaptureConfig,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    return {
        "original_filename": candidate.path.name,
        "extension": candidate.path.suffix.lower(),
        "size_bytes": int(candidate.stat.st_size),
        "modified_at": _iso_modified_at(candidate.stat),
        "source_dir_display": config.capture_dir_display,
        "age_seconds": _age_seconds(candidate.stat, now=now),
    }


def _is_stale(candidate: CaptureCandidate, config: ScannerCaptureConfig, *, now: float | None = None) -> bool:
    return _age_seconds(candidate.stat, now=now) > config.max_age_seconds


def get_scanner_status() -> dict[str, Any]:
    config = load_scanner_capture_config()
    saved_provider = SavedFileBridgeProvider(config)
    twain_provider = TwainHelperProvider(config)
    saved_status = saved_provider.status()
    twain_status = twain_provider.status()
    saved_available = bool(saved_status["available"])
    twain_available = bool(twain_status["available"])
    twain_source_detected = bool(twain_status.get("source_detected"))
    available_modes = []
    if twain_available:
        available_modes.append("twain")
    if saved_available:
        available_modes.append("saved_file_bridge")

    if twain_available:
        active_mode = "twain"
    elif saved_available:
        active_mode = "saved_file_bridge"
    else:
        active_mode = "unavailable"

    last_error = None
    if not twain_available and not saved_available:
        last_error = twain_status.get("last_error") or NO_CAPTURE_MESSAGE
    elif not twain_available:
        last_error = twain_status.get("last_error")

    latest_metadata = saved_status.get("latest_capture")
    return {
        "active_mode": active_mode,
        "available_modes": available_modes,
        "direct_capture_available": twain_available,
        "saved_file_bridge_available": saved_available,
        "device_detected": None,
        "device_name": BIOMETRIKA_TWAIN_SOURCE_NAME if twain_source_detected else None,
        "driver_detected": twain_source_detected,
        "twain_source_detected": twain_source_detected,
        "umpi_cli_available": False,
        "sdk_dlls_detected": _sdk_dlls_detected(),
        "last_error": last_error,
        "diagnostics": {
            "twain": twain_status,
            "saved_file_bridge": saved_status,
            "twain_raw_dir_display": config.twain_raw_dir_display,
        },
        "configured": True,
        "enabled": twain_available or saved_available,
        "direct_capture_enabled": twain_available,
        "saved_file_bridge_enabled": saved_available,
        "capture_dir_display": config.capture_dir_display,
        "normalized_dir_display": config.normalized_dir_display,
        "capture_glob": config.capture_glob,
        "max_age_seconds": config.max_age_seconds,
        "latest_capture": latest_metadata,
        "latest_capture_is_stale": saved_status.get("latest_capture_is_stale"),
    }


def _short_hash(candidate: CaptureCandidate) -> str:
    digest = hashlib.sha256()
    digest.update(candidate.path.name.encode("utf-8", errors="surrogatepass"))
    digest.update(str(candidate.stat.st_mtime_ns).encode("ascii"))
    digest.update(str(candidate.stat.st_size).encode("ascii"))
    with candidate.path.open("rb") as handle:
        digest.update(handle.read(65536))
    return digest.hexdigest()[:8]


def _capture_id(candidate: CaptureCandidate) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"scanner_{timestamp}_{_short_hash(candidate)}"


def _normalize_image_to_png(source: Path, destination: Path) -> None:
    if Image is None or ImageOps is None:  # pragma: no cover - depends on optional runtime package
        raise ScannerCaptureNormalizationError("Pillow is required to normalize scanner captures to PNG.")

    try:
        with Image.open(source) as image:
            try:
                image.seek(0)
            except EOFError:
                pass
            image = ImageOps.exif_transpose(image)
            image.load()

            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                normalized = image.convert("RGBA")
            elif image.mode in {"1", "L", "I", "I;16", "F"}:
                normalized = image.convert("L")
            else:
                normalized = image.convert("RGB")

            normalized.save(destination, format="PNG")
    except UnidentifiedImageError as exc:
        raise ScannerCaptureNormalizationError("Unable to read saved UMPI capture image.") from exc
    except OSError as exc:
        raise ScannerCaptureNormalizationError(f"Unable to normalize saved UMPI capture image: {exc}") from exc


def _normalized_file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "format": "png",
        "mime_type": "image/png",
    }


def _raw_file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "format": path.suffix.lower().lstrip("."),
    }


def _normalize_candidate(
    candidate: CaptureCandidate,
    config: ScannerCaptureConfig,
    *,
    source_error_message: str,
) -> dict[str, Any]:
    config.normalized_dir.mkdir(parents=True, exist_ok=True)
    capture_id = _capture_id(candidate)
    normalized_filename = f"{capture_id}.png"
    normalized_path = config.normalized_dir / normalized_filename
    try:
        _normalize_image_to_png(candidate.path, normalized_path)
    except ScannerCaptureNormalizationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive normalization wrapper
        raise ScannerCaptureNormalizationError(source_error_message) from exc

    metadata = _capture_metadata(candidate, config)
    return {
        "capture_id": capture_id,
        **metadata,
        "normalized_filename": normalized_filename,
        "normalized_url": f"/api/scanner/captures/{capture_id}",
        "mime_type": "image/png",
        "raw_path": str(candidate.path),
        "normalized_path": str(normalized_path),
        "raw_file": _raw_file_metadata(candidate.path),
        "normalized_file": _normalized_file_metadata(normalized_path),
    }


def import_latest_capture() -> dict[str, Any]:
    config = load_scanner_capture_config()
    latest = find_latest_capture(config)
    if latest is None:
        raise NoScannerCaptureFoundError(NO_CAPTURE_MESSAGE)
    if _is_stale(latest, config):
        raise StaleScannerCaptureError(STALE_CAPTURE_MESSAGE)

    result = _normalize_candidate(
        latest,
        config,
        source_error_message="Unable to normalize saved UMPI capture image.",
    )
    return {
        key: value
        for key, value in result.items()
        if key not in {"raw_path", "normalized_path", "raw_file", "normalized_file"}
    }


def _is_windows() -> bool:
    return os.name == "nt"


def _path_is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _run_twain_helper(args: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    run_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "check": False,
        "timeout": timeout_seconds,
    }
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", None)
    if create_no_window is not None:
        run_kwargs["creationflags"] = create_no_window
    return subprocess.run(args, **run_kwargs)


def _parse_helper_json(stdout: str, *, stderr: str | None = None) -> dict[str, Any]:
    raw = stdout.strip()
    if not raw:
        raise ScannerCaptureProviderError(
            "invalid_helper_output",
            "TWAIN helper produced no JSON on stdout.",
            diagnostics={"stderr": (stderr or "").strip()},
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScannerCaptureProviderError(
            "invalid_helper_output",
            "TWAIN helper stdout was not valid JSON.",
            diagnostics={"stdout": raw[:1000], "stderr": (stderr or "").strip()},
        ) from exc
    if not isinstance(payload, dict):
        raise ScannerCaptureProviderError(
            "invalid_helper_output",
            "TWAIN helper JSON must be an object.",
            diagnostics={"stdout": raw[:1000]},
        )
    return payload


def _map_helper_capture_error(payload: dict[str, Any], default: str = "helper_failed") -> str:
    helper_code = str(payload.get("error_code") or "")
    if helper_code == "timeout":
        return "capture_timeout"
    if helper_code in {"twain_unavailable", "source_not_found"}:
        return "twain_unavailable"
    return default


def _validate_helper_output_path(output_path: Any, expected_output_dir: Path) -> Path:
    if not isinstance(output_path, str) or not output_path.strip():
        raise ScannerCaptureProviderError(
            "invalid_helper_output",
            "TWAIN helper did not return an output_path.",
            diagnostics={"output_path": output_path},
        )

    output_root = expected_output_dir.resolve(strict=True)
    try:
        raw_path = Path(output_path).resolve(strict=True)
    except OSError as exc:
        raise ScannerCaptureProviderError(
            "invalid_helper_output",
            "TWAIN helper output_path does not exist.",
            diagnostics={"output_path": output_path},
        ) from exc

    if not _path_is_inside(raw_path, output_root):
        raise ScannerCaptureProviderError(
            "invalid_helper_output",
            "TWAIN helper output_path was outside the expected output directory.",
            diagnostics={"output_path": str(raw_path), "expected_output_dir": str(output_root)},
        )
    if raw_path.suffix.lower() not in VALID_HELPER_EXTENSIONS:
        raise ScannerCaptureProviderError(
            "invalid_helper_output",
            "TWAIN helper output_path had an unsupported image extension.",
            diagnostics={"output_path": str(raw_path)},
        )
    if Image is None:
        raise ScannerCaptureProviderError(
            "invalid_helper_output",
            "Pillow is required to validate TWAIN helper output.",
            diagnostics={"output_path": str(raw_path)},
        )

    try:
        with Image.open(raw_path) as image:
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ScannerCaptureProviderError(
            "invalid_helper_output",
            "TWAIN helper output_path was not a readable image.",
            diagnostics={"output_path": str(raw_path)},
        ) from exc

    return raw_path


def _sdk_dlls_detected() -> bool | None:
    vendor_root = REPO_ROOT / "external" / "biometrika"
    if not vendor_root.exists():
        return None
    return any(path.is_file() for path in vendor_root.rglob("*.dll"))


class SavedFileBridgeProvider:
    name = "saved_file_bridge"

    def __init__(self, config: ScannerCaptureConfig) -> None:
        self.config = config

    def status(self) -> dict[str, Any]:
        latest = find_latest_capture(self.config)
        latest_metadata = _capture_metadata(latest, self.config) if latest else None
        return {
            "available": self.config.capture_dir.is_dir(),
            "capture_dir_display": self.config.capture_dir_display,
            "normalized_dir_display": self.config.normalized_dir_display,
            "capture_glob": self.config.capture_glob,
            "max_age_seconds": self.config.max_age_seconds,
            "latest_capture": latest_metadata,
            "latest_capture_is_stale": _is_stale(latest, self.config) if latest else None,
        }

    def capture(self, *, normalize: bool = True) -> dict[str, Any]:
        start = time.monotonic()
        latest = find_latest_capture(self.config)
        if latest is None:
            raise ScannerCaptureProviderError("fallback_unavailable", NO_CAPTURE_MESSAGE)
        if _is_stale(latest, self.config):
            raise ScannerCaptureProviderError("fallback_unavailable", STALE_CAPTURE_MESSAGE)

        if normalize:
            try:
                result = _normalize_candidate(
                    latest,
                    self.config,
                    source_error_message="Unable to normalize saved UMPI capture image.",
                )
            except ScannerCaptureNormalizationError as exc:
                raise ScannerCaptureProviderError("fallback_unavailable", str(exc)) from exc
        else:
            capture_id = _capture_id(latest)
            result = {
                "capture_id": capture_id,
                **_capture_metadata(latest, self.config),
                "normalized_filename": None,
                "normalized_url": None,
                "mime_type": None,
                "raw_path": str(latest.path),
                "normalized_path": None,
                "raw_file": _raw_file_metadata(latest.path),
                "normalized_file": None,
            }

        return {
            "ok": True,
            "mode_used": self.name,
            "direct_capture": False,
            "normalized_url": result["normalized_url"],
            "capture_id": result["capture_id"],
            "raw_file": result["raw_file"],
            "normalized_file": result["normalized_file"],
            "duration_ms": int((time.monotonic() - start) * 1000),
            "device": {"name": None, "provider": self.name},
            "warning": None,
            "metadata": {
                "original_filename": result["original_filename"],
                "modified_at": result["modified_at"],
                "age_seconds": result["age_seconds"],
                "source_dir_display": result["source_dir_display"],
            },
        }


class TwainHelperProvider:
    name = "twain"

    def __init__(self, config: ScannerCaptureConfig) -> None:
        self.config = config

    def status(self) -> dict[str, Any]:
        helper_path = self.config.twain_helper_path
        diagnostics: dict[str, Any] = {
            "helper_path": str(helper_path),
            "helper_path_display": self.config.twain_helper_path_display,
            "windows": _is_windows(),
        }
        if not _is_windows():
            return {
                "available": False,
                "source_detected": False,
                "last_error": "TWAIN direct capture is only available on Windows.",
                "payload": None,
                "diagnostics": {**diagnostics, "stage": "platform"},
            }
        if not helper_path.is_file():
            return {
                "available": False,
                "source_detected": False,
                "last_error": "TWAIN helper executable is missing.",
                "payload": None,
                "diagnostics": {**diagnostics, "stage": "helper_missing"},
            }

        try:
            completed = _run_twain_helper([str(helper_path), "--status"], timeout_seconds=8.0)
            payload = _parse_helper_json(completed.stdout, stderr=completed.stderr)
        except subprocess.TimeoutExpired:
            return {
                "available": False,
                "source_detected": False,
                "last_error": "TWAIN helper status timed out.",
                "payload": None,
                "diagnostics": {**diagnostics, "stage": "status_timeout"},
            }
        except OSError as exc:
            return {
                "available": False,
                "source_detected": False,
                "last_error": "TWAIN helper could not be executed.",
                "payload": None,
                "diagnostics": {
                    **diagnostics,
                    "stage": "status_subprocess_oserror",
                    "error": str(exc),
                },
            }
        except ScannerCaptureProviderError as exc:
            return {
                "available": False,
                "source_detected": False,
                "last_error": exc.message,
                "payload": None,
                "diagnostics": {**diagnostics, "stage": "status_parse", **exc.diagnostics},
            }

        source_detected = bool(payload.get("ok") is True and payload.get("source_detected") is True)
        available = completed.returncode == 0 and source_detected
        return {
            "available": available,
            "source_detected": source_detected,
            "last_error": None if available else str(payload.get("message") or "TWAIN source is not available."),
            "payload": payload,
            "diagnostics": {
                **diagnostics,
                "stage": "status",
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
            },
        }

    def capture(
        self,
        *,
        timeout_ms: int = 15000,
        show_ui: bool = False,
        normalize: bool = True,
        settle_after_enable_ms: int = DEFAULT_HEADLESS_SETTLE_AFTER_ENABLE_MS,
    ) -> dict[str, Any]:
        start = time.monotonic()
        if not _is_windows():
            raise ScannerCaptureProviderError("twain_unavailable", "TWAIN direct capture is only available on Windows.")
        if not self.config.twain_helper_path.is_file():
            raise ScannerCaptureProviderError("twain_unavailable", "TWAIN helper executable is missing.")

        try:
            self.config.twain_raw_dir.mkdir(parents=True, exist_ok=True)
            output_dir = Path(tempfile.mkdtemp(prefix="capture_", dir=str(self.config.twain_raw_dir))).resolve(
                strict=True
            )
        except OSError as exc:
            raise ScannerCaptureProviderError(
                "helper_failed",
                "Could not create TWAIN helper output directory.",
                diagnostics={
                    "stage": "create_twain_output_dir",
                    "error": str(exc),
                    "twain_raw_dir": str(self.config.twain_raw_dir),
                },
            ) from exc
        args = [
            str(self.config.twain_helper_path),
            "--capture",
            "--output-dir",
            str(output_dir),
            "--show-ui",
            "true" if show_ui else "false",
            "--timeout-ms",
            str(timeout_ms),
            "--settle-after-enable-ms",
            str(settle_after_enable_ms),
        ]

        try:
            completed = _run_twain_helper(args, timeout_seconds=(timeout_ms / 1000.0) + 5.0)
        except subprocess.TimeoutExpired as exc:
            raise ScannerCaptureProviderError(
                "capture_timeout",
                "TWAIN helper capture timed out.",
                diagnostics={"stage": "subprocess_timeout", "timeout_ms": timeout_ms},
            ) from exc
        except OSError as exc:
            raise ScannerCaptureProviderError(
                "helper_failed",
                "TWAIN helper could not be executed.",
                diagnostics={
                    "stage": "capture_subprocess_oserror",
                    "error": str(exc),
                },
            ) from exc

        payload = _parse_helper_json(completed.stdout, stderr=completed.stderr)
        if completed.returncode != 0 or payload.get("ok") is not True:
            error_code = _map_helper_capture_error(payload)
            raise ScannerCaptureProviderError(
                error_code,
                str(payload.get("message") or "TWAIN helper capture failed."),
                diagnostics={
                    "returncode": completed.returncode,
                    "helper": payload,
                    "stderr": completed.stderr.strip(),
                },
            )

        raw_path = _validate_helper_output_path(payload.get("output_path"), output_dir)
        raw_candidate = CaptureCandidate(path=raw_path, stat=raw_path.stat())
        if normalize:
            try:
                result = _normalize_candidate(
                    raw_candidate,
                    self.config,
                    source_error_message="Unable to normalize TWAIN helper output image.",
                )
            except ScannerCaptureNormalizationError as exc:
                raise ScannerCaptureProviderError(
                    "invalid_helper_output",
                    str(exc),
                    diagnostics={"output_path": str(raw_path)},
                ) from exc
        else:
            capture_id = _capture_id(raw_candidate)
            result = {
                "capture_id": capture_id,
                **_capture_metadata(raw_candidate, self.config),
                "normalized_filename": None,
                "normalized_url": None,
                "mime_type": None,
                "raw_path": str(raw_path),
                "normalized_path": None,
                "raw_file": _raw_file_metadata(raw_path),
                "normalized_file": None,
        }

        helper_duration = payload.get("duration_ms")
        duration_ms = (
            int(helper_duration)
            if isinstance(helper_duration, (int, float))
            else int((time.monotonic() - start) * 1000)
        )
        return {
            "ok": True,
            "mode_used": self.name,
            "direct_capture": True,
            "normalized_url": result["normalized_url"],
            "capture_id": result["capture_id"],
            "raw_file": result["raw_file"],
            "normalized_file": result["normalized_file"],
            "duration_ms": duration_ms,
            "device": {"name": BIOMETRIKA_TWAIN_SOURCE_NAME, "provider": self.name},
            "warning": None,
            "helper": payload,
        }


def _failure_response(
    *,
    error_code: str,
    message: str,
    mode_requested: str,
    fallback_available: bool,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": error_code,
        "message": message,
        "mode_requested": mode_requested,
        "fallback_available": fallback_available,
        "diagnostics": diagnostics or {},
    }


def _capture_saved_with_response(
    saved_provider: SavedFileBridgeProvider,
    *,
    normalize: bool,
    warning: str | None = None,
) -> dict[str, Any]:
    result = saved_provider.capture(normalize=normalize)
    result["warning"] = warning
    return result


def _clamp_settle_after_enable_ms(value: int) -> int:
    return max(MIN_SETTLE_AFTER_ENABLE_MS, min(MAX_SETTLE_AFTER_ENABLE_MS, int(value)))


def _resolve_settle_after_enable_ms(value: int | None, *, show_ui: bool) -> int:
    if value is None:
        return 0 if show_ui else DEFAULT_HEADLESS_SETTLE_AFTER_ENABLE_MS
    return _clamp_settle_after_enable_ms(value)


def capture_scanner(
    *,
    mode: str = "auto",
    timeout_ms: int = 15000,
    fallback_allowed: bool = False,
    normalize: bool = True,
    show_ui: bool = False,
    settle_after_enable_ms: int | None = None,
) -> dict[str, Any]:
    config = load_scanner_capture_config()
    saved_provider = SavedFileBridgeProvider(config)
    twain_provider = TwainHelperProvider(config)
    saved_status = saved_provider.status()
    fallback_available = bool(saved_status["available"])
    resolved_settle_after_enable_ms = _resolve_settle_after_enable_ms(
        settle_after_enable_ms,
        show_ui=show_ui,
    )

    if mode not in {"auto", "twain", "saved_file_bridge"}:
        return _failure_response(
            error_code="no_capture_available",
            message="Unsupported scanner capture mode.",
            mode_requested=mode,
            fallback_available=fallback_available,
            diagnostics={"mode": mode},
        )

    if mode == "saved_file_bridge":
        try:
            return _capture_saved_with_response(saved_provider, normalize=normalize)
        except ScannerCaptureProviderError as exc:
            return _failure_response(
                error_code=exc.error_code,
                message=exc.message,
                mode_requested=mode,
                fallback_available=fallback_available,
                diagnostics=exc.diagnostics,
            )

    twain_status = twain_provider.status()
    twain_available = bool(twain_status["available"])
    fallback_warning = "TWAIN direct capture failed/unavailable; saved-file bridge fallback was used."

    if not twain_available:
        if fallback_allowed and fallback_available:
            try:
                return _capture_saved_with_response(saved_provider, normalize=normalize, warning=fallback_warning)
            except ScannerCaptureProviderError as exc:
                return _failure_response(
                    error_code="fallback_unavailable",
                    message=exc.message,
                    mode_requested=mode,
                    fallback_available=fallback_available,
                    diagnostics={"twain": twain_status, "fallback": exc.diagnostics},
                )
        error_code = "twain_unavailable" if mode in {"auto", "twain"} else "no_capture_available"
        return _failure_response(
            error_code=error_code,
            message="TWAIN direct capture is unavailable and fallback is not allowed.",
            mode_requested=mode,
            fallback_available=fallback_available,
            diagnostics={"twain": twain_status},
        )

    try:
        return twain_provider.capture(
            timeout_ms=timeout_ms,
            show_ui=show_ui,
            normalize=normalize,
            settle_after_enable_ms=resolved_settle_after_enable_ms,
        )
    except ScannerCaptureProviderError as exc:
        if fallback_allowed and fallback_available:
            try:
                return _capture_saved_with_response(saved_provider, normalize=normalize, warning=fallback_warning)
            except ScannerCaptureProviderError as fallback_exc:
                return _failure_response(
                    error_code="fallback_unavailable",
                    message=fallback_exc.message,
                    mode_requested=mode,
                    fallback_available=fallback_available,
                    diagnostics={"twain": exc.diagnostics, "fallback": fallback_exc.diagnostics},
                )
        return _failure_response(
            error_code=exc.error_code,
            message=exc.message,
            mode_requested=mode,
            fallback_available=fallback_available,
            diagnostics=exc.diagnostics,
        )


def resolve_normalized_capture_path(capture_id: str, config: ScannerCaptureConfig | None = None) -> Path:
    if not CAPTURE_ID_PATTERN.fullmatch(capture_id):
        raise ScannerCapturePathError("Invalid scanner capture id.")

    resolved_config = config or load_scanner_capture_config()
    normalized_root = resolved_config.normalized_dir.resolve(strict=False)
    filename = capture_id if capture_id.endswith(".png") else f"{capture_id}.png"
    candidate = (normalized_root / filename).resolve(strict=False)

    try:
        candidate.relative_to(normalized_root)
    except ValueError as exc:
        raise ScannerCapturePathError("Invalid scanner capture path.") from exc

    if candidate.suffix.lower() != ".png":
        raise ScannerCapturePathError("Scanner capture must be a PNG file.")
    if not candidate.is_file():
        raise ScannerNormalizedCaptureNotFoundError("Scanner capture not found.")

    return candidate
