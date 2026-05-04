from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import os
from pathlib import Path
import re
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
DEFAULT_CAPTURE_GLOB = "*.tif;*.tiff;*.png;*.bmp;*.jpg;*.jpeg"
DEFAULT_MAX_AGE_SECONDS = 3600

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}
CAPTURE_ID_PATTERN = re.compile(r"^scanner_\d{8}_\d{6}_[a-f0-9]{8}(?:\.png)?$")

NO_CAPTURE_MESSAGE = "No saved scanner capture found in the configured folder."
STALE_CAPTURE_MESSAGE = "Latest saved UMPI capture is too old. Save a new fingerprint scan and try again."


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


@dataclass(frozen=True)
class ScannerCaptureConfig:
    capture_dir: Path
    normalized_dir: Path
    capture_glob: str
    max_age_seconds: int
    capture_dir_display: str
    normalized_dir_display: str


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
    capture_glob = os.getenv("SCANNER_CAPTURE_GLOB", DEFAULT_CAPTURE_GLOB).strip() or DEFAULT_CAPTURE_GLOB

    return ScannerCaptureConfig(
        capture_dir=capture_dir,
        normalized_dir=normalized_dir,
        capture_glob=capture_glob,
        max_age_seconds=_configured_max_age(),
        capture_dir_display=_display_path(capture_dir, capture_dir_raw),
        normalized_dir_display=_display_path(normalized_dir, normalized_dir_raw),
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
    latest = find_latest_capture(config)
    latest_metadata = _capture_metadata(latest, config) if latest else None

    return {
        "configured": True,
        "enabled": config.capture_dir.is_dir(),
        "capture_dir_display": config.capture_dir_display,
        "normalized_dir_display": config.normalized_dir_display,
        "capture_glob": config.capture_glob,
        "max_age_seconds": config.max_age_seconds,
        "latest_capture": latest_metadata,
        "latest_capture_is_stale": _is_stale(latest, config) if latest else None,
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


def import_latest_capture() -> dict[str, Any]:
    config = load_scanner_capture_config()
    latest = find_latest_capture(config)
    if latest is None:
        raise NoScannerCaptureFoundError(NO_CAPTURE_MESSAGE)
    if _is_stale(latest, config):
        raise StaleScannerCaptureError(STALE_CAPTURE_MESSAGE)

    config.normalized_dir.mkdir(parents=True, exist_ok=True)
    capture_id = _capture_id(latest)
    normalized_filename = f"{capture_id}.png"
    normalized_path = config.normalized_dir / normalized_filename
    _normalize_image_to_png(latest.path, normalized_path)

    metadata = _capture_metadata(latest, config)
    return {
        "capture_id": capture_id,
        **metadata,
        "normalized_filename": normalized_filename,
        "normalized_url": f"/api/scanner/captures/{capture_id}",
        "mime_type": "image/png",
    }


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
